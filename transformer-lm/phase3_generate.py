import sys
sys.path.insert(0, "/mnt/14T/houyi/slm/transformer-lm")

import torch
from lm_basics.transformer import TransformerLM
from lm_basics.decode import generate
from lm_basics.tokenizer import Tokenizer

tok = Tokenizer.from_files("data/vocab.json", "data/merges.txt", ["<|endoftext|>"])

model = TransformerLM(vocab_size=10000, context_length=512, d_model=512,
                      num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0)
ckpt = torch.load("save/phase2c_main_step4999", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompts = [
    ("Once upon a time,", "once-upon-a-time"),
    ("The little girl",   "the-little-girl"),
    ("I went to the",     "i-went-to-the"),
    ("He looked at",      "he-looked-at"),
    ("She said,",         "she-said"),
]

for prompt_text, tag in prompts:
    prompt_ids = tok.encode(prompt_text)
    ids = generate(model, prompt_ids, max_new_tokens=256, temperature=0.8, top_p=0.9)
    text = tok.decode(ids)
    print(f"{'='*60}")
    print(f"PROMPT: {prompt_text}")
    print(f"LEN: {len(ids)} tokens")
    print(f"GEN:")
    print(text)
    print()
