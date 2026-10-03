import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(model_id_or_dir: str, device: str, attn_implementation: str | None = None):
    if attn_implementation is None:
        attn_implementation = "eager" if device == "cpu" else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    if tokenizer.pad_token is None:
        # Llama-3.1's tokenizer ships without a pad token, and everything that
        # batches variable-length pairs -- `tokenize_prompt_and_output`, DPO, the
        # HF rollout backend -- pads to the longest one. Without this they die
        # inside `torch.full(..., pad_id)` with a None pad id.
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
