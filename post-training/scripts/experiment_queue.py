"""Keep experiments running: start the next queued job whenever GPUs free up.

The box is shared and every card usually has somebody's job on it, so by default the
scheduler **shares**: it takes a card as soon as it has enough free memory, without
waiting for one to go completely idle. It never kills anything, and it never asks
for more than the run needs -- vLLM preallocates `--gpu-memory-utilization` of the
card, so the request is kept small and the leftover is left to whoever is already
there. `--exclusive` restores the strict rule (only cards with no compute processes
at all), which on this box means waiting indefinitely.

A card also has to look free for `--settle-polls` consecutive polls before the
scheduler commits, so a card that is merely between two of somebody else's jobs is
not mistaken for an available one.

The floor is per *role*, not per card, but the two roles are closer than they look.
Measured on this box: the training process of a GRPO job peaks at **44GB** (sampled
live while an 8.5GB neighbour sat on the card), and the rollout server occupies
**up to 38GB** -- not the ~15GB its `--gpu-memory-utilization 0.3` budget suggests.
Its KV pool is only 11.8GB (`/metrics`: 5624 blocks x 16 tokens for OLMo-1B), and
the footprint moves around (18GB early in a run, 23-38GB later), so it is not
bounded by the budget. A server with no training loop attached matches its budget
exactly, so the extra is tied to the weight-sync path -- mechanism unidentified,
worth its own look (a track4 question).

So `fits()` applies the job's own floor to the first (most free) card and the
scheduler's global floor to the rest, and both floors are set from these numbers.
Demanding the *training* floor from both cards would be nearly the same thing here,
but not quite: a 41GB card can serve as a rollout card and not as a training one.

Jobs run in priority order (track1 > track2 > track3). The scheduler starts the
first pending job whose GPU requirement it can satisfy, and never more than
`--max-concurrent` at once (default 1 -- taking four cards at once is rude).

Restarting is safe: a job whose run directory already holds `final/` is marked
done and skipped, and a job still marked running is checked against its pid.

    setsid nohup .venv/bin/python scripts/experiment_queue.py > runs/_queue/queue.log 2>&1 < /dev/null &
    .venv/bin/python scripts/experiment_queue.py --status
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
STATE_PATH = REPO_ROOT / "runs" / "_queue" / "state.json"
LOG_DIR = REPO_ROOT / "runs" / "_queue"

# The standard run everything else is compared against. Zero-shot `r1_zero` is not
# usable here: the base model scores 0.08% on it, so roughly 99% of groups would
# be all-wrong and carry no gradient at all -- 17 GPU-hours of nothing.
THREE_SHOT = "r1_zero_three_shot_gsm8k"

# vLLM preallocates a fraction of TOTAL memory, and the box is shared, so keep the
# request small; 0.3 of a 48GB card left room for E3 alongside other users.
COMMON_EXTRA = ("--vllm-gpu-memory-utilization", "0.3")

# `run_experiments.py` disables wandb when these are empty, so a run launched
# without them silently lands nowhere -- which is exactly what happened to the
# first queued job. E3 passed them by hand; every job has to.
WANDB_PROJECT = "reasoning-rl"
WANDB_ENTITY = "houyi"


@dataclass(frozen=True)
class Job:
    name: str
    argv: list[str]
    gpus: int = 2
    note: str = ""
    # How to tell the job is done. `checkpoints/` is written *inside* the run dir,
    # and `checkpoints/final/` is the last thing a successful GRPO run writes.
    # `steps` is the fallback: a row count in metrics.jsonl.
    marker: str = ""
    steps: int = 0
    # The floor for the card this job *trains* on; the other cards of a two-card
    # job only have to clear the scheduler's global floor. See TRAIN_MIN_FREE_MIB.
    min_free_mib: int = 0
    # Continue from the run's newest checkpoint instead of starting over. Only
    # honoured when that checkpoint actually exists (see `resumable`), and only
    # correct because a fixed seed plus the restored optimizer/RNG state makes a
    # resumed run identical to an uninterrupted one -- this saves the steps
    # already paid for, it does not change the experiment.
    resume: bool = False

    @property
    def run_dir(self) -> Path:
        return REPO_ROOT / "runs" / self.name


# What the *training* card of a GRPO job needs. Measured, not guessed: a run
# allocated 36GB and then asked for 3 more on a card that also held somebody's
# 8.5GB, and OOMed -- three times; the live one then peaked at 44.08GB. The peak
# tracks the longest response in the batch, so the floor has to be "room for
# ~45GB plus a neighbour", not the average.
TRAIN_MIN_FREE_MIB = 45_000


def grpo(name: str, variant: str, seed: int, steps: int, *, prompt: str = THREE_SHOT,
         flags: tuple[str, ...] = (), note: str = "", resume: bool = False) -> Job:
    return Job(
        name=name,
        gpus=2,
        note=note,
        marker=f"runs/{name}/checkpoints/final",
        steps=steps,
        min_free_mib=TRAIN_MIN_FREE_MIB,
        resume=resume,
        argv=[
            sys.executable, str(SCRIPTS / "run_experiments.py"),
            "--variant", variant, "--seed", str(seed),
            "--num-rollout-steps", str(steps), "--prompt", prompt,
            "--output-dir", f"runs/{name}",
            "--train-gpu", "{gpu0}", "--vllm-gpu", "{gpu1}", "--vllm-port", "{port}",
            "--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY,
            *flags,
            "--extra-args", " ".join(COMMON_EXTRA),
        ],
    )


def panorama(name: str, *, limit: int, samples: int, note: str = "") -> Job:
    return Job(
        name=name,
        gpus=1,
        note=note,
        marker=f"runs/{name}/samples.jsonl",
        argv=[
            sys.executable, str(SCRIPTS / "tts_panorama.py"),
            "--prompt", THREE_SHOT, "--limit", str(limit), "--num-samples", str(samples),
            "--gpu", "{gpu0}", "--port", "{port}",
            "--gpu-memory-utilization", "0.15",
            "--output", f"runs/{name}/samples.jsonl",
        ],
    )


QUEUE: list[Job] = [
    # ---------------- track1 (priority 1) ----------------
    # E3 seed 0 is running under runs/e3_3shot_seed0; this is its pair. E3 is the
    # only experiment kept at 200 steps -- everything else is 80.
    grpo("e3_3shot_seed1", "grpo_sequence", 1, 200, note="E3 seed 1"),

    # E6: on-policy variants. 3-shot rather than the zero-shot `r1_zero` the
    # reference recipe names, for the reason in THREE_SHOT above; E6 has to keep
    # hyperparameters fixed against the standard run, which is 3-shot.
    *[grpo(f"e6_{variant}_seed{seed}", variant, seed, 80, note="E6")
      for variant in ("grpo_constant", "dr_grpo", "rft", "maxrl")
      for seed in (0, 1)],

    # E4: learning-rate sweep around the 1e-5 default. The midpoint doubles as the
    # 80-step standard reference the other 80-step runs get compared against.
    *[grpo(f"e4_lr{lr:g}_seed0", "grpo_sequence", 0, 80,
           flags=("--learning-rate", f"{lr:g}"), note="E4")
      for lr in (3e-6, 1e-5, 3e-5)],

    # E5: prompt ablation. Its 3-shot arm is the e4_lr1e-05 run above; the
    # zero-shot arm it is compared against is E1/E2.
    *[grpo(f"e5_question_only_seed{seed}", "grpo_sequence", seed, 80,
           prompt="question_only", note="E5")
      for seed in (0, 1)],

    # E7: off-policy.
    *[grpo(f"e7_{variant}_seed{seed}", variant, seed, 80, note="E7")
      for variant in ("offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo")
      for seed in (0, 1)],

    # ---------------- track2 (priority 2) ----------------
    # V1: what the base model already reaches with repeated sampling. Single card.
    panorama("track2_v1_panorama", limit=512, samples=32, note="V1"),

    # V2: the same run with a weak verifier swapped in. The `exact` arm is E3
    # truncated to 80 steps -- which only holds because nothing else changes.
    *[grpo(f"track2_{variant}_seed{seed}", variant, seed, 80, note="V2")
      for variant in ("veritas_format_only", "veritas_noisy0.1")
      for seed in (0, 1)],
    # The noisy0.3 runs are the ones this box keeps OOM-ing: the flip noise stops
    # groups from being all-right or all-wrong, so almost nothing is pruned and
    # `n_sequences_kept` sits at 256/256 -- the largest training step on the board.
    # They resume from their newest checkpoint after a neighbour takes the card.
    *[grpo(f"track2_veritas_noisy0.3_seed{seed}", "veritas_noisy0.3", seed, 80,
           note="V2", resume=True)
      for seed in (0, 1)],

    # ---------------- follow-up (priority 4) ----------------
    # E5's question_only arm barely moved in 80 steps: reward sat near 0 and the
    # format rate stayed flat, because the base model emits `\boxed{}` only ~17% of
    # the time, so ~98% of groups are all-wrong and carry no gradient. A 200-step
    # extension was queued to ask whether 80 is simply too few for a prompt this
    # weak, and dropped on 2026-10-03 (user's call): the 80-step pair already
    # carries the point and ~5h of cards buys nothing for the write-up. The state
    # file keeps the entry marked `skipped`, so it does not look like it vanished.
]

# track3's SFT and DPO are deliberately **not** in here. They were, and it cost
# more than it saved: a manual SFT registered as a running job would occupy the
# concurrency slot, and a manual DPO got its run directory parked out from under
# it by the queue's own retry. Two one-off training runs do not need a scheduler
# -- launch them by hand with the same env the queue uses (see `launch`).


@dataclass
class Gpu:
    free_mib: int
    procs: int


@dataclass
class State:
    status: dict[str, str] = field(default_factory=dict)     # name -> pending/running/done/failed
    pid: dict[str, int] = field(default_factory=dict)
    started: dict[str, float] = field(default_factory=dict)
    finished: dict[str, float] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    gpus: dict[str, list[int]] = field(default_factory=dict)

    def save(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(
            {"status": self.status, "pid": self.pid, "started": self.started,
             "finished": self.finished, "attempts": self.attempts, "gpus": self.gpus},
            indent=2,
        ))

    @classmethod
    def load(cls) -> "State":
        if not STATE_PATH.exists():
            return cls()
        raw = json.loads(STATE_PATH.read_text())
        return cls(status=raw["status"], pid=raw.get("pid", {}),
                   started=raw.get("started", {}), finished=raw.get("finished", {}),
                   attempts=raw.get("attempts", {}), gpus=raw.get("gpus", {}))


def last_used_gpus(state: State) -> list[int]:
    """The cards the most recently started job took."""
    if not state.started:
        return []
    name = max(state.started, key=lambda key: state.started[key])
    return state.gpus.get(name, [])


def settle(state: State, job: Job, max_retries: int) -> str:
    """Decide what a job that is no longer running becomes.

    A run that produced `final/` is done. Anything else is retried up to
    `max_retries` times first -- most failures on this box are transient (a
    neighbour took the memory we were sharing, a port was grabbed), and a queue
    that gives up on the first hiccup is not much of a queue.
    """
    if finished(job):
        return "done"
    if state.attempts.get(job.name, 0) < max_retries:
        state.attempts[job.name] = state.attempts.get(job.name, 0) + 1
        return "pending"
    return "failed"


def _smi(query: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line.split(", ") for line in out.strip().splitlines() if line.strip()]


def probe_gpus() -> dict[int, Gpu]:
    """index -> free memory and compute-process count."""
    index_of = {uuid: int(index) for index, uuid in _smi("gpu=index,uuid")}
    gpus = {int(index): Gpu(free_mib=int(free), procs=0) for index, free in _smi("gpu=index,memory.free")}
    for uuid, _pid in _smi("compute-apps=gpu_uuid,pid"):
        index = index_of.get(uuid)
        if index is not None:
            gpus[index].procs += 1
    return gpus


def free_port(preferred: int, taken: set[int]) -> int | None:
    for port in range(preferred, preferred + 60):
        if port in taken:
            continue
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return None


def counted_steps(run_dir: Path) -> int:
    """Parsed rows in metrics.jsonl; a torn last line (kill mid-write) is skipped."""
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return 0
    rows = 0
    for line in metrics.read_text(encoding="utf-8").splitlines():
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        rows += 1
    return rows


def finished(job: Job) -> bool:
    if job.marker and (REPO_ROOT / job.marker).exists():
        return True
    if job.steps:
        return counted_steps(job.run_dir) >= job.steps
    return False


def release_checkpoints(job: Job) -> None:
    """Drop the periodic checkpoints once a run is in the books, keep the policy.

    `checkpoints/stepNNNN/` exists for exactly one purpose, `--resume`, and a
    finished run is never resumed. It is also the bulk of the disk: ~9GB of Adam
    moments against 2.9GB of weights. `checkpoints/final/` -- the trained policy,
    saved without optimizer state -- stays, so the model can still be loaded.
    Across the 29 queued jobs this is ~250GB.
    """
    checkpoints = job.run_dir / "checkpoints"
    if not (checkpoints / "final").is_dir():
        return
    freed = 0.0
    for entry in checkpoints.iterdir():
        if entry.is_dir() and entry.name != "final":
            freed += sum(f.stat().st_size for f in entry.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(entry)
    if freed:
        typer.echo(f"           released {job.run_dir.name}/checkpoints/step* ({freed:.1f} GB)")


def park_previous_run(job: Job, attempt: int) -> None:
    """Move an unfinished run dir aside so a launch can start clean.

    `RunLogger` refuses to write into a directory that already holds metrics, and
    a job only gets here after dying without `final/`. Renaming rather than
    deleting keeps the evidence for whoever asks why it died -- which matters,
    because the first two attempts at anything tend to fail for a reason that is
    worth reading once.
    """
    if not job.run_dir.exists() or finished(job):
        return
    candidate = job.run_dir.with_name(f"{job.run_dir.name}.attempt{attempt}")
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = job.run_dir.with_name(f"{job.run_dir.name}.attempt{attempt}-{suffix}")
    job.run_dir.rename(candidate)
    typer.echo(f"           parked {job.run_dir.name} -> {candidate.name}")


def resumable(job: Job) -> bool:
    """Whether the launch should continue this job's run instead of starting over.

    Only with the job's own consent, and only when the newest checkpoint holds a
    `training_state.pt` -- `grpo_train.py --resume` refuses a directory whose
    metrics have no checkpoint beside them, and that guard is worth keeping: it is
    what stops a fresh run from silently appending onto a dead one's log.
    """
    if not job.resume:
        return False
    checkpoints = job.run_dir / "checkpoints"
    if not checkpoints.is_dir():
        return False
    return any(
        path.name.startswith("step") and (path / "training_state.pt").is_file()
        for path in checkpoints.iterdir()
    )


def launch(job: Job, gpus: list[int], port: int, resume: bool = False) -> subprocess.Popen:
    argv = [arg.format(gpu0=gpus[0], gpu1=gpus[-1], port=port) for arg in job.argv]
    if resume:
        # `--resume` goes through `--extra-args`, which run_experiments.py forwards
        # to grpo_train.py verbatim; on its own it would be an unknown option to
        # run_experiments.py itself.
        index = argv.index("--extra-args") + 1
        argv[index] = f"{argv[index]} --resume".strip()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOG_DIR / f"{job.name}.log").open("w")
    log.write(f"$ {' '.join(argv)}\n\n")
    log.flush()
    # vLLM is launched as a CLI ("vllm serve") by vllm_utils.start_server, and the
    # entry point lives in the venv -- without this on PATH every job dies at once
    # with FileNotFoundError: 'vllm'.
    # This box also needs P2P disabled, or the NCCL weight sync spins forever at
    # ~96% CPU and never reaches the server.
    venv_bin = str(REPO_ROOT / ".venv" / "bin")
    env = {
        **os.environ,
        "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "NCCL_P2P_DISABLE": "1",
        # GRPO's training card OOMs with 3-4GB sitting "reserved by PyTorch but
        # unallocated" -- fragmentation, not real demand. This lets the allocator
        # hand those segments back instead of failing the allocation.
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    return subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def describe(state: State) -> None:
    for index, job in enumerate(QUEUE, start=1):
        status = state.status.get(job.name, "pending")
        extra = ""
        if status == "done":
            extra = f"  ({(state.finished[job.name] - state.started[job.name]) / 3600:.1f}h)"
        elif status == "running":
            extra = f"  pid {state.pid.get(job.name)}"
        typer.echo(f"{index:>2}. {job.name:<34} {status:<8}{job.note:<8}{extra}")


@app.command()
def main(
    poll_seconds: int = typer.Option(60, help="Seconds between GPU probes."),
    min_free_mib: int = typer.Option(
        40_000,
        help="Free memory a card needs to be considered at all, and the floor for the "
             "card of a two-card job that does not train (it hosts the rollout server). "
             "That server needs ~38GB, not the ~15GB its 0.3 utilization suggests -- see "
             "the module docstring -- so this floor is nearly the training one. The card "
             "a job trains on must clear the job's own floor on top of this.",
    ),
    exclusive: bool = typer.Option(
        False,
        help="Only take cards with no compute processes at all. Off by default: on this box "
             "almost every card carries somebody's job, so that means waiting indefinitely.",
    ),
    settle_polls: int = typer.Option(2, help="Consecutive free polls before a card is trusted."),
    max_concurrent: int = typer.Option(1, help="Jobs to run at once."),
    max_retries: int = typer.Option(
        2, help="Times to relaunch a job that died without producing final/. A queue that "
                "gives up on the first hiccup is not much of a queue, but this one should not "
                "loop forever either.",
    ),
    port_base: int = typer.Option(8300, help="First port to try for vLLM."),
    max_jobs: int = typer.Option(0, help="Stop after this many launches; 0 = run the whole queue."),
    status: bool = typer.Option(False, "--status", help="Print the queue and exit."),
    dry_run: bool = typer.Option(False, help="Probe GPUs, print the next launch, and exit."),
) -> None:
    state = State.load()
    for job in QUEUE:
        state.status.setdefault(job.name, "done" if finished(job) else "pending")

    # A job we recorded as running may have died with the previous scheduler.
    for name, pid in list(state.pid.items()):
        if state.status.get(name) != "running" or alive(pid):
            continue
        job = next((j for j in QUEUE if j.name == name), None)
        if job is None:
            # Dropped from the queue since the state was last written (track3's SFT
            # and DPO were removed this way). Nothing left to reconcile it against,
            # so stop tracking it instead of raising StopIteration on startup.
            state.status[name] = "done"
            continue
        state.status[name] = settle(state, job, max_retries)
        state.finished.setdefault(name, time.time())

    if status:
        describe(state)
        return

    state.save()
    typer.echo(f"queue: {len(QUEUE)} jobs, "
               f"{sum(1 for j in QUEUE if state.status[j.name] == 'pending')} pending")

    running: dict[str, subprocess.Popen | None] = {}
    ports: dict[str, int] = {}
    for name, pid in state.pid.items():
        if state.status.get(name) == "running":
            running[name] = None  # adopted; polled by pid in the loop
    streak: dict[int, int] = {}   # consecutive polls each GPU has looked free
    launches = 0

    def stop(signum, _frame):
        typer.echo(f"\nsignal {signum}: marking state and exiting (children keep running)")
        state.save()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while True:
        for name in list(running):
            process = running[name]
            done = (process is None and not alive(state.pid[name])) or (
                process is not None and process.poll() is not None)
            if not done:
                continue
            job = next(j for j in QUEUE if j.name == name)
            elapsed = (time.time() - state.started.get(name, time.time())) / 3600
            state.status[name] = settle(state, job, max_retries)
            state.finished[name] = time.time()
            running.pop(name)
            ports.pop(name, None)
            state.save()
            typer.echo(f"[{time.strftime('%H:%M:%S')}] {name} -> {state.status[name]} ({elapsed:.1f}h)")
            if state.status[name] == "done":
                release_checkpoints(job)

        pending = [j for j in QUEUE if state.status[j.name] == "pending"]
        more = bool(pending) and not (max_jobs and launches >= max_jobs)
        if not more and not running:
            typer.echo("queue drained")
            break

        # Probe and report on every poll, even when every concurrency slot is taken.
        # A scheduler whose log goes quiet while a job runs cannot be told apart
        # from one that has died -- and this one sat silent for nine hours with
        # eighteen jobs queued, because the whole block used to sit inside the
        # "can I start something" branch.
        gpus = probe_gpus()
        for index, gpu in gpus.items():
            roomy = gpu.free_mib >= min_free_mib and (not exclusive or gpu.procs == 0)
            if roomy:
                streak[index] = streak.get(index, 0) + 1
            else:
                streak[index] = 0
        # Most room first: among cards that qualify, take the one we intrude on
        # least, not the lowest index.
        ready = sorted(
            (index for index, count in streak.items() if count >= settle_polls),
            key=lambda index: -gpus[index].free_mib,
        )
        # The previous job's cards are the first choice, and they skip the settle
        # window: they were ours a minute ago, so there is nothing left to be
        # suspicious of. Re-probing from scratch would also cost two polls of dead
        # time at every handoff.
        carried = [
            index for index in last_used_gpus(state)
            if index in gpus and gpus[index].free_mib >= min_free_mib
        ]
        # Otherwise most room first, so we intrude on the neighbour least.
        usable = carried + [index for index in ready if index not in carried]
        # Never hand out a card one of our own jobs is already using. Without this
        # the scheduler will start a run alongside, say, a manual SFT that was
        # registered as a running job, and both fight for the memory.
        busy = {index for name in running for index in state.gpus.get(name, [])}
        usable = [index for index in usable if index not in busy]
        typer.echo(f"[{time.strftime('%H:%M:%S')}] free: "
                   + ", ".join(f"{i}({gpus[i].free_mib // 1024}G)" for i in usable)
                   + f" | running {len(running)} | pending {len(pending)}")

        if more and len(running) < max_concurrent:

            def fits(job: Job) -> list[int]:
                """Cards this job can use, most free first.

                Only the first card trains, so only the first card has to clear the
                job's own floor; the rest just have to clear the global one (they
                host a vLLM server). See the module docstring.
                """
                floor = max(min_free_mib, job.min_free_mib)
                roomy = [index for index in usable if gpus[index].free_mib >= floor]
                if not roomy:
                    return []
                rest = [index for index in usable if index != roomy[0]]
                return roomy[:1] + rest

            job = next((j for j in pending if len(fits(j)) >= j.gpus), None)
            if dry_run:
                typer.echo(f"would start: {job.name if job else '(nothing fits)'}")
                return
            if job is not None:
                port = free_port(port_base, set(ports.values()))
                if port is None:
                    typer.echo("           every port in range is taken; waiting")
                else:
                    chosen = fits(job)[:job.gpus]
                    resume_now = resumable(job)
                    if not resume_now:
                        # Nothing to continue from (or the job did not ask to), so
                        # hand the launch a clean directory.
                        park_previous_run(job, state.attempts.get(job.name, 0))
                    process = launch(job, chosen, port, resume=resume_now)
                    running[job.name] = process
                    ports[job.name] = port
                    state.gpus[job.name] = chosen
                    state.status[job.name] = "running"
                    state.pid[job.name] = process.pid
                    state.started[job.name] = time.time()
                    launches += 1
                    state.save()
                    typer.echo(f"           started {job.name} on GPU {chosen}, port {port}")

        if dry_run:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    app()
