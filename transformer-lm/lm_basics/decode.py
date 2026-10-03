"""
Text decoding / generation for a trained Transformer LM.

Supports temperature scaling and top-p (nucleus) sampling.
"""
import torch

from .transformer import softmax


def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """Sample a single token from (possibly temperature-scaled, top-p filtered) logits.

    Args:
        logits:  (vocab_size,)  unnormalised logits for the next token
        temperature: > 0  flattens (<1) or sharpens (>1) the distribution.  0.0 → greedy argmax.
        top_p:      0 - 1   keep only the smallest set of tokens whose cumulative prob ≥ top_p

    Returns:
        scalar LongTensor — the sampled token id
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=False)

    scaled = logits / temperature
    probs = softmax(scaled, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)                         # prefix-sum, ascending by prob
        mask = cumsum - sorted_probs >= top_p                       # after position where cumsum crosses p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()

        # Gather back into vocabulary order and sample
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

    # Guard against zero-probability distribution (can happen in pathological top-p)
    if probs.sum() <= 0:
        return logits.argmax(dim=-1, keepdim=False)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """Autoregressively sample a continuation from a Transformer LM.

    Args:
        model:           TransformerLM instance
        prompt:          list of token ids to start from
        max_new_tokens:  stop after generating this many tokens
        temperature:     softmax temperature (0.0 = greedy argmax)
        top_p:           nucleus threshold (1.0 = no filtering)
        eos_token_id:    stop early if this token is generated (e.g. <|endoftext|>)
        device:          torch device string

    Returns:
        list[int] — the full sequence (prompt + generated continuation)
    """
    model.eval()
    generated = list(prompt)

    for _ in range(max_new_tokens):
        x = torch.tensor([generated], dtype=torch.long, device=device)   # (1, seq)
        logits = model(x)                                                 # (1, seq, vocab)
        next_logits = logits[0, -1, :]                                    # (vocab,)

        next_token = sample_token(next_logits, temperature=temperature, top_p=top_p)
        token_id = int(next_token.item())
        generated.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

    return generated
