"""GRPO training loop for GSM8K.

`--rollout-backend vllm` needs two GPUs: NCCL will not put two ranks on one
device, so the policy and the vLLM server must be separate cards. `hf` samples
with the policy itself, which fits on one GPU at about half the speed.

Smoke test:
    CUDA_VISIBLE_DEVICES=3 python scripts/grpo_train.py --rollout-backend hf \
        --n-train 32 --n-val 16 --num-rollout-steps 4 --rollout-batch-size 8 \
        --group-size 8 --gradient-accumulation-steps 1 --output-dir runs/smoke
"""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import typer

from lm_alignment.checkpoint import get_model_and_tokenizer
from lm_alignment.data import build_prompts, load_gsm8k, load_prompt_template
from lm_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from lm_alignment.weak_verifiers import format_only_reward_fn, make_noisy_reward_fn
from lm_alignment.evaluation import compute_group_pass_at_k, grade_rollouts, summarize_rollouts
from lm_alignment.grpo import get_response_log_probs, grpo_train_step
from lm_alignment.run_logging import RunLogger, build_config
from lm_alignment.tokenization import tokenize_prompt_and_output
from lm_alignment.vllm_utils import VLLMServer

app = typer.Typer(add_completion=False)


class CachingRewardFn:
    """Memoize grading: grpo_train_step grades the same rollouts again, via sympy."""

    def __init__(self, reward_fn, cache: dict[tuple[str, str], dict[str, float]]):
        self._reward_fn = reward_fn
        self._cache = cache

    def __call__(self, response: str, ground_truth: str) -> dict[str, float]:
        key = (response, ground_truth)
        if key not in self._cache:
            self._cache[key] = self._reward_fn(response, ground_truth)
        return self._cache[key]


def sampling_logprob_difference(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    rollouts: list[Rollout],
    device: torch.device,
    batch_size: int = 8,
) -> tuple[float, float] | None:
    """Mean and max gap between the sampling policy's log-probs and the trainer's.

    ~0 when the engine is in sync, and the only metric that catches a stale one,
    because the importance ratio reads 1.0 regardless. None if not requested.
    """
    if not rollouts or not rollouts[0].sampling_log_probs:
        return None

    tokenized = tokenize_prompt_and_output(prompts, [r.text for r in rollouts], tokenizer)
    input_ids, labels, mask = tokenized["input_ids"], tokenized["labels"], tokenized["response_mask"]

    deltas: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            stop = min(start + batch_size, input_ids.shape[0])
            chunk_mask = mask[start:stop]
            trainer_log_probs = get_response_log_probs(
                model=model,
                input_ids=input_ids[start:stop].to(device),
                labels=labels[start:stop].to(device),
                return_token_entropy=False,
            )["log_probs"].float().cpu()
            # mask is per-row contiguous, so one flattened gather plus running
            # offsets recovers each row's response tokens in order.
            flat = trainer_log_probs[chunk_mask]
            offset = 0
            for row_index, rollout in enumerate(rollouts[start:stop]):
                count = int(chunk_mask[row_index].sum())
                ours = flat[offset : offset + count]
                offset += count
                sampled = rollout.sampling_log_probs or []
                # Re-tokenizing the decoded text can differ from the ids the
                # server used; compare the shared prefix rather than guessing.
                count = min(count, len(sampled))
                if count:
                    deltas.append((ours[:count] - torch.tensor(sampled[:count])).abs())

    if not deltas:
        return None
    stacked = torch.cat(deltas)
    return stacked.mean().item(), stacked.max().item()


def build_sampling_params(
    temperature: float, max_tokens: int, n: int, seed: int, stop_on_answer: bool
) -> dict:
    params = {"temperature": temperature, "max_tokens": max_tokens, "n": n, "seed": seed}
    if stop_on_answer:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


@dataclass
class Rollout:
    text: str
    token_ids: list[int]
    # "length" means it hit max_tokens; the HF backend leaves this None.
    finish_reason: str | None = None
    # Log-probabilities the *sampling* policy assigned to its own tokens. Only
    # available when the server is asked for them.
    sampling_log_probs: list[float] | None = None


class RolloutSource:
    """Rollouts from vLLM, or from the policy itself for a single-GPU run.

    vLLM needs a second GPU: NCCL will not put two ranks on one device.
    """

    def __init__(
        self,
        backend: str,
        policy: torch.nn.Module,
        tokenizer,
        device: torch.device,
        server: VLLMServer | None = None,
        batch_size: int = 8,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.tokenizer = tokenizer
        self.device = device
        self.server = server
        self.batch_size = batch_size

    def generate(
        self,
        prompts: list[str],
        group_size: int,
        temperature: float,
        max_tokens: int,
        seed: int,
        stop_on_answer: bool,
    ) -> list[Rollout]:
        if self.backend == "vllm":
            # Sync here on purpose. As a separate call it gets forgotten, and the
            # failure is invisible: the importance ratio reads 1.0 regardless.
            self.server.sync_policy_weights(self.policy)
            sampling_params = build_sampling_params(
                temperature, max_tokens, group_size, seed, stop_on_answer
            )
            # Needed by sampling_logprob_difference, the stale-engine check.
            sampling_params["logprobs"] = 0
            # One prompt per request keeps the n completions grouped per prompt,
            # but generate_completions issues its requests in a loop, so it also
            # serializes them. Training needs the grouping (n = group_size);
            # evaluation is n=1, where every choice has index 0 and the sort is
            # stable, so batching there is safe and turns 1024 round-trips into 16.
            request_batch_size = 1 if group_size > 1 else 64
            completions = self.server.generate_completions(
                prompts, sampling_params, batch_size=request_batch_size
            )
            return [
                Rollout(
                    text=c.text,
                    token_ids=c.token_ids,
                    finish_reason=c.finish_reason,
                    sampling_log_probs=c.token_log_probs,
                )
                for c in completions
            ]
        return self._generate_with_policy(prompts, group_size, temperature, max_tokens, stop_on_answer)

    @torch.no_grad()
    def _generate_with_policy(
        self,
        prompts: list[str],
        group_size: int,
        temperature: float,
        max_tokens: int,
        stop_on_answer: bool,
    ) -> list[Rollout]:
        was_training = self.policy.training
        padding_side = self.tokenizer.padding_side
        # Generation needs left padding: with right padding a short prompt's
        # generated tokens would start inside its own pad region.
        self.tokenizer.padding_side = "left"
        self.policy.eval()
        try:
            rollouts = []
            for start in range(0, len(prompts), self.batch_size):
                chunk = prompts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    chunk, return_tensors="pt", padding=True, add_special_tokens=False
                )
                input_ids = encoded["input_ids"].to(self.device)
                # vLLM reads 0 as greedy; HF raises on it, and eval defaults to 0.
                do_sample = temperature > 0
                generated = self.policy.generate(
                    input_ids=input_ids,
                    attention_mask=encoded["attention_mask"].to(self.device),
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=1.0,
                    max_new_tokens=max_tokens,
                    # generate expands each prompt into `group_size` rows in
                    # order, matching the repeated-prompt layout the RL loop uses.
                    num_return_sequences=group_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stop_strings=["</answer>"] if stop_on_answer else None,
                    tokenizer=self.tokenizer if stop_on_answer else None,
                )
                for row in generated[:, input_ids.shape[1] :]:
                    token_ids = row.tolist()
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    if stop_on_answer and "</answer>" in text:
                        text = text[: text.index("</answer>") + len("</answer>")]
                    rollouts.append(Rollout(text=text, token_ids=token_ids))
            return rollouts
        finally:
            self.policy.train(was_training)
            self.tokenizer.padding_side = padding_side


@torch.no_grad()
def compute_old_log_probs(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
    chunk_size: int,
) -> list[torch.Tensor]:
    """Old log-probs, one tensor per training chunk.

    `tokenize_prompt_and_output` pads to the longest pair in whatever batch it is
    handed. Tokenizing the whole rollout batch in one call therefore yields columns
    padded to the batch maximum, while the training step re-tokenizes a single
    chunk and pads to *that* chunk's maximum -- and the two no longer subtract:
    "size of tensor a (720) must match the size of tensor b (1041)". Tokenizing per
    chunk here keeps both paddings identical. Only the off-policy path reads these
    (importance reweighting), which is why `offpolicy_naive` never tripped it.

    Still one pass over the whole rollout batch, before any gradient step: these
    values have to come from the sampling policy, not from a partly-trained one.
    """
    chunks = []
    with torch.no_grad():
        for start in range(0, len(prompts), chunk_size):
            tokenized = tokenize_prompt_and_output(
                prompts[start : start + chunk_size], responses[start : start + chunk_size], tokenizer
            )
            scores = get_response_log_probs(
                model=model,
                input_ids=tokenized["input_ids"].to(device),
                labels=tokenized["labels"].to(device),
                return_token_entropy=False,
            )
            # `.float().cpu()` is differentiable, so without `no_grad` the result
            # carries a grad_fn and keeps this chunk's whole autograd graph alive
            # -- for every one of the 256 sequences, all the way through the
            # training steps that follow. Nothing here is ever backwarded.
            chunks.append(scores["log_probs"].float().cpu())
    return chunks


def evaluate(
    source: RolloutSource,
    examples: list[dict[str, str]],
    template: str,
    reward_fn,
    temperature: float,
    max_tokens: int,
    seed: int,
    stop_on_answer: bool,
    reward_workers: int,
) -> tuple[dict[str, float], list[str], list[Rollout], list[str]]:
    """One completion per validation prompt, greedy by default.

    Sampling is noisier and would make the curve reflect the temperature.
    """
    prompts = build_prompts(template, [e["question"] for e in examples])
    rollouts = source.generate(
        prompts,
        group_size=1,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        stop_on_answer=stop_on_answer,
    )
    if len(rollouts) != len(examples):
        raise RuntimeError(f"Expected {len(examples)} completions, got {len(rollouts)}.")
    responses = [r.text for r in rollouts]
    ground_truths = [e["ground_truth"] for e in examples]
    metrics = summarize_rollouts(
        grade_rollouts(responses, ground_truths, reward_fn, num_workers=reward_workers)
    )
    metrics["response_len_mean"] = sum(len(r.token_ids) for r in rollouts) / len(rollouts)
    return metrics, prompts, rollouts, ground_truths


def capture_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Resume the sampler where it left off, so a resumed run isn't a fresh draw."""
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    save_training_state: bool = True,
) -> None:
    """Weights plus what `--resume` needs.

    `save_training_state=False` skips the optimizer state, which is ~2x the weights.
    """
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_directory=str(path))
    tokenizer.save_pretrained(save_directory=str(path))
    if save_training_state:
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng_state(),
            },
            path / "training_state.pt",
        )
    typer.echo(f"saved checkpoint to {path}")


def load_training_state(path: Path, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, dict]:
    """Restore optimizer momentum (Adam's second moment matters) and RNG state."""
    state = torch.load(path / "training_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    rng = state.get("rng", {})
    # `map_location=device` moves the RNG tensors to the GPU along with everything
    # else, and both `torch.set_rng_state` and `torch.cuda.set_rng_state_all` want
    # *CPU* uint8 tensors. Handed a CUDA one they say "RNG state must be a
    # torch.ByteTensor" -- which blames the dtype, not the device, and reads as
    # though the checkpoint were corrupt. Back to the CPU they go.
    if isinstance(rng.get("torch"), torch.Tensor):
        rng["torch"] = rng["torch"].cpu()
    if isinstance(rng.get("cuda"), list):
        rng["cuda"] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in rng["cuda"]]
    return state["step"], rng


def prune_checkpoints(checkpoint_root: Path, keep: int) -> None:
    """Keep only the newest `keep` step checkpoints; disk here is 95% full."""
    if keep <= 0:
        return
    existing = sorted(checkpoint_root.glob("step[0-9]*"))
    for stale in existing[:-keep]:
        shutil.rmtree(stale)
        typer.echo(f"removed old checkpoint {stale.name}")


def write_rollouts(path: Path, prompts, completions, ground_truths) -> None:
    # zip() would truncate to the shortest, which silently mislabels every
    # rollout after the first prompt when the lists are not the same length.
    if not (len(prompts) == len(completions) == len(ground_truths)):
        raise ValueError(
            f"rollout dump needs aligned lists, got {len(prompts)} prompts, "
            f"{len(completions)} completions, {len(ground_truths)} ground truths"
        )
    with path.open("w", encoding="utf-8") as f:
        for prompt, completion, truth in zip(prompts, completions, ground_truths):
            text = completion.text if hasattr(completion, "text") else str(completion)
            f.write(
                f"{'=' * 80}\nPROMPT:\n{prompt}\n{'-' * 80}\n"
                f"RESPONSE:\n{text}\n{'-' * 80}\nGROUND TRUTH: {truth}\n"
            )


def build_reward_fn(name: str, prompt: str, seed: int):
    """Resolve `--reward-fn` into the callable the training loop grades with.

    'auto' keeps the original behaviour, so existing runs are unaffected.
    """
    if name == "auto":
        return question_only_reward_fn if prompt == "question_only" else r1_zero_reward_fn
    if name == "format-only":
        return format_only_reward_fn
    if name.startswith("noisy:"):
        return make_noisy_reward_fn(r1_zero_reward_fn, epsilon=float(name.split(":", 1)[1]), seed=seed)
    if name == "self-verify":
        raise typer.BadParameter(
            "self-verify needs a call back into an inference engine; not implemented yet."
        )
    raise typer.BadParameter(f"Unknown --reward-fn {name!r}.")


@app.command()
def main(
    model: str = typer.Option("allenai/OLMo-2-0425-1B", help="Model path or HF id."),
    train_path: str = typer.Option("data/gsm8k/train.jsonl"),
    val_path: str = typer.Option("data/gsm8k/test.jsonl"),
    prompt: str = typer.Option("r1_zero", help="Prompt template name in lm_alignment/prompts/."),
    output_dir: str = typer.Option("runs/grpo"),
    n_train: int = typer.Option(6400),
    n_val: int = typer.Option(1024),
    num_rollout_steps: int = typer.Option(200),
    rollout_batch_size: int = typer.Option(256),
    train_batch_size: int = typer.Option(256),
    group_size: int = typer.Option(8),
    gradient_accumulation_steps: int = typer.Option(32),
    learning_rate: float = typer.Option(1e-5),
    max_grad_norm: float = typer.Option(1.0),
    sampling_temperature: float = typer.Option(1.0),
    sampling_max_tokens: int = typer.Option(512),
    eval_temperature: float = typer.Option(0.0, help="0 = greedy."),
    baseline: str = typer.Option("mean"),
    advantage_normalizer: str = typer.Option("std"),
    loss_normalization: str = typer.Option("sequence"),
    normalization_constant: int | None = typer.Option(None),
    importance_reweighting_method: str = typer.Option("none"),
    cliprange: float | None = typer.Option(None),
    eval_every: int = typer.Option(10),
    log_rollouts_every: int = typer.Option(40),
    checkpoints_every: int = typer.Option(
        25, help="Steps between checkpoints; 0 disables them and with it --resume."
    ),
    keep_checkpoints: int = typer.Option(
        1, help="How many step checkpoints to retain; 0 keeps all. Each is ~9GB."
    ),
    resume: bool = typer.Option(
        False, help="Continue the run in --output-dir from its newest checkpoint."
    ),
    rollout_backend: str = typer.Option(
        "vllm",
        help="'vllm' is fast but needs a second GPU for the server (NCCL weight sync "
        "requires two distinct devices); 'hf' samples with the policy itself on one GPU.",
    ),
    hf_rollout_batch_size: int = typer.Option(8, help="Prompts per generate() call for --rollout-backend hf."),
    policy_device: str = typer.Option("cuda:0", help="Device for the trainable policy."),
    vllm_gpu: int = typer.Option(1, help="Physical GPU index for the vLLM server."),
    vllm_port: int = typer.Option(
        8000, help="Must differ per concurrent run: startup kills other servers on this port."
    ),
    vllm_gpu_memory_utilization: float = typer.Option(0.85),
    attn_implementation: str = typer.Option(
        "sdpa", help="flash_attention_2 requires the flash-attn package."
    ),
    reward_workers: int = typer.Option(16),
    reward_fn_name: str = typer.Option(
        "auto",
        "--reward-fn",
        help="'auto' picks by prompt (question_only vs r1_zero). 'format-only' and "
        "'noisy:<eps>' swap in a weak verifier (see lm_alignment/weak_verifiers.py); "
        "everything else about the run is unchanged, so the exact arm stays comparable.",
    ),
    seed: int = typer.Option(0),
    overwrite_run: bool = typer.Option(
        False, help="Reuse the output dir, discarding the previous run's metrics."
    ),
    tensorboard: bool = typer.Option(True, help="Write event files under <run-dir>/tensorboard."),
    wandb_project: str | None = typer.Option(None),
    wandb_entity: str | None = typer.Option(None, help="Your wandb user or team; required if the account has no default."),
) -> None:
    run_config = build_config(locals())
    if rollout_backend not in ("vllm", "hf"):
        raise typer.BadParameter("rollout_backend must be 'vllm' or 'hf'.")
    if rollout_batch_size % group_size != 0:
        raise typer.BadParameter("rollout_batch_size must be a multiple of group_size.")
    if train_batch_size % group_size != 0:
        raise typer.BadParameter("train_batch_size must be a multiple of group_size.")
    if rollout_batch_size % train_batch_size != 0:
        raise typer.BadParameter("rollout_batch_size must be a multiple of train_batch_size.")
    steps_per_rollout = rollout_batch_size // train_batch_size
    if importance_reweighting_method != "none" and steps_per_rollout == 1:
        raise typer.BadParameter(
            "Off-policy importance reweighting needs several train steps per rollout batch."
        )

    random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    device = torch.device(policy_device)

    # Created before anything else so console.log captures startup too.
    logger = RunLogger(
        run_dir=out_dir,
        config=run_config,
        tensorboard=tensorboard,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        overwrite=overwrite_run,
        resume=resume,
    )
    resume_checkpoint = logger.latest_checkpoint() if resume else None
    if resume and resume_checkpoint is None:
        # --resume bypasses RunLogger's overwrite guard, so this is the only
        # thing stopping it from appending a fresh run onto the old metrics.
        raise typer.BadParameter(
            f"--resume given but {out_dir}/checkpoints holds no usable checkpoint. "
            f"Pass --overwrite-run to start over, or delete {out_dir}."
        )
    start_step = 0

    train_examples = load_gsm8k(train_path, limit=n_train)
    val_examples = load_gsm8k(val_path, limit=n_val)
    template = load_prompt_template(prompt)
    reward_fn = build_reward_fn(reward_fn_name, prompt, seed)
    # Validation always scores with the **exact** grader, whatever the training
    # reward is. A weak verifier's verdict is the thing under study, not the
    # measurement of it: the format-only arms passed their training reward here and
    # `val_accuracy` read 0.0 for all 80 steps -- that was the format rate of a
    # correct answer, not its accuracy, and every weak-verifier curve was fake.
    eval_reward_fn = question_only_reward_fn if prompt == "question_only" else r1_zero_reward_fn
    stop_on_answer = prompt != "question_only"
    typer.echo(f"train={len(train_examples)} val={len(val_examples)} prompt={prompt}")

    policy, tokenizer = get_model_and_tokenizer(
        str(resume_checkpoint) if resume_checkpoint else model,
        policy_device,
        attn_implementation=attn_implementation,
    )
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    if resume_checkpoint is not None:
        start_step, rng_state = load_training_state(resume_checkpoint, optimizer, device)
        restore_rng_state(rng_state)
        start_step += 1
        typer.echo(f"resumed from {resume_checkpoint.name}; continuing at step {start_step}")

    server = None
    if rollout_backend == "vllm":
        server = VLLMServer(
            model_id=model,
            port=vllm_port,
            gpu=vllm_gpu,
            seed=seed,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            log_fileno=logger.console_fileno,
        )
        server.start()
        server.init_weight_sync(policy_device)
    source = RolloutSource(
        backend=rollout_backend,
        policy=policy,
        tokenizer=tokenizer,
        device=device,
        server=server,
        batch_size=hf_rollout_batch_size,
    )

    n_prompts_per_batch = rollout_batch_size // group_size
    start_time = time.time()

    for step in range(start_step, num_rollout_steps):
        batch = random.sample(train_examples, n_prompts_per_batch)
        unique_prompts = build_prompts(template, [e["question"] for e in batch])
        unique_truths = [e["ground_truth"] for e in batch]
        repeated_prompts = [p for p in unique_prompts for _ in range(group_size)]
        repeated_truths = [t for t in unique_truths for _ in range(group_size)]

        generate_start = time.time()
        completions = source.generate(
            unique_prompts,
            group_size=group_size,
            temperature=sampling_temperature,
            max_tokens=sampling_max_tokens,
            seed=seed + step,
            stop_on_answer=stop_on_answer,
        )
        generate_s = time.time() - generate_start
        responses = [c.text for c in completions]
        if len(responses) != len(repeated_prompts):
            raise RuntimeError(f"Expected {len(repeated_prompts)} completions, got {len(responses)}.")

        grade_start = time.time()
        scores = grade_rollouts(responses, repeated_truths, reward_fn, num_workers=reward_workers)
        grade_s = time.time() - grade_start

        # A response cut off at max_tokens scores 0 for reasons unrelated to policy.
        clipped = [
            (c.finish_reason == "length")
            if c.finish_reason is not None
            else len(c.token_ids) >= sampling_max_tokens
            for c in completions
        ]
        response_lengths = [len(c.token_ids) for c in completions]
        sampling_gap = sampling_logprob_difference(
            policy, tokenizer, repeated_prompts, completions, device
        )
        cached_reward_fn = CachingRewardFn(
            reward_fn, {(r, g): s for r, g, s in zip(responses, repeated_truths, scores)}
        )
        raw_rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)

        old_log_probs = None
        if importance_reweighting_method != "none":
            old_log_probs = compute_old_log_probs(
                policy, tokenizer, repeated_prompts, responses, device, train_batch_size
            )

        train_start = time.time()
        step_metrics = []
        for chunk in range(steps_per_rollout):
            lo, hi = chunk * train_batch_size, (chunk + 1) * train_batch_size
            _, train_metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                reward_fn=cached_reward_fn,
                repeated_prompts=repeated_prompts[lo:hi],
                rollout_responses=responses[lo:hi],
                repeated_ground_truths=repeated_truths[lo:hi],
                group_size=group_size,
                baseline=baseline,
                advantage_normalizer=advantage_normalizer,
                importance_reweighting_method=importance_reweighting_method,
                old_log_probs=None if old_log_probs is None else old_log_probs[chunk],
                cliprange=cliprange,
                loss_normalization=loss_normalization,
                normalization_constant=normalization_constant,
            )
            step_metrics.append({k: float(v) for k, v in train_metadata.items()})
        train_s = time.time() - train_start

        # Average every metric the step reported, not a hand-picked few. Keys are
        # *not* identical across chunks: a chunk whose every advantage was pruned
        # runs none of `grpo_train_step`'s microbatches and reports no loss metrics
        # at all. Averaging over `step_metrics[0]`'s keys then raised KeyError on
        # whichever chunk had been emptied -- which only the off-policy runs, with
        # their 32 small chunks, ever hit.
        reported = {key for metrics in step_metrics for key in metrics}
        train_means = {
            key: sum(m[key] for m in step_metrics if key in m)
            / sum(1 for m in step_metrics if key in m)
            for key in reported
        }
        record = {
            **train_means,
            "train_accuracy": summarize_rollouts(scores)["accuracy"],
            "train_response_len_mean": sum(response_lengths) / len(response_lengths),
            "train_response_len_max": max(response_lengths),
            "response_clipped_ratio": sum(clipped) / len(clipped),
            **compute_group_pass_at_k(raw_rewards, group_size),
            # Where the wall clock actually goes; generation usually dominates and
            # you cannot tune what you have not measured.
            "time_generate_s": generate_s,
            "time_grade_s": grade_s,
            "time_train_s": train_s,
        }
        if sampling_gap is not None:
            record["sampling_logprob_gap_mean"], record["sampling_logprob_gap_max"] = sampling_gap

        if step % eval_every == 0 or step == num_rollout_steps - 1:
            val_metrics, val_prompts, val_rollouts, val_truths = evaluate(
                source,
                val_examples,
                template,
                eval_reward_fn,
                eval_temperature,
                sampling_max_tokens,
                seed,
                stop_on_answer,
                reward_workers,
            )
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            # Greedy validation has low variance and reflects deployment, but it
            # is not comparable to the prompting baselines, which sample at
            # temperature 1.0. Log both so the report can
            # use each for what it is good for.
            if sampling_temperature != eval_temperature:
                sampled_metrics, _, _, _ = evaluate(
                    source,
                    val_examples,
                    template,
                    eval_reward_fn,
                    sampling_temperature,
                    sampling_max_tokens,
                    seed,
                    stop_on_answer,
                    reward_workers,
                )
                record.update(
                    {f"val_sampled_{k}": v for k, v in sampled_metrics.items()}
                )
            if step == 0:
                write_rollouts(
                    logger.rollout_path("val_step0"), val_prompts, val_rollouts, val_truths
                )

        record["elapsed_s"] = time.time() - start_time
        logger.log(step, record)
        typer.echo(f"step {step}: { {k: round(v, 4) for k, v in record.items()} }")

        if log_rollouts_every and step > 0 and step % log_rollouts_every == 0:
            # Repeated, not unique: `completions` has group_size entries per
            # prompt, and zipping it against the unique prompts would silently
            # pair each prompt with some other prompt's rollout.
            write_rollouts(
                logger.rollout_path(f"train_step{step:04d}"),
                repeated_prompts,
                completions,
                repeated_truths,
            )
        if checkpoints_every and step > 0 and step % checkpoints_every == 0:
            save_checkpoint(
                policy, tokenizer, optimizer, step, logger.checkpoint_dir(f"step{step:04d}")
            )
            prune_checkpoints(out_dir / "checkpoints", keep_checkpoints)

    # `final` skips the optimizer state: nothing ever resumes from it, and it
    # would double the on-disk cost of every run.
    save_checkpoint(
        policy,
        tokenizer,
        optimizer,
        num_rollout_steps - 1,
        logger.checkpoint_dir("final"),
        save_training_state=False,
    )
    logger.close()
    typer.echo(f"done in {time.time() - start_time:.0f}s; artifacts in {out_dir}")


if __name__ == "__main__":
    app()
