"""
Resource accounting for a Transformer LM (transformer_accounting)
+ training-time estimates for A100 / A6000.

All numbers assume the architecture used in this project (SwiGLU FFN with d_ff ≈ 8/3·d_model,
RMSNorm, no bias).  Embedding lookup is a gather (zero FLOPs for accounting purposes).

FLOPs rule:  A ∈ ℝ^{m×n}, B ∈ ℝ^{n×p}  ⇒  AB costs 2·m·n·p FLOPs
(n multiplies + n additions per entry, m·p entries)
"""
from __future__ import annotations
import math

# ── GPU peak FP16 tensor TFLOPS ─────────────────────────────────────────────
A100_F16  = 312.0     # A100 SXM 80 GB
A6000_F16 = 77.4      # A6000 (no sparsity)
B200_F16  = 450.0     # Blackwell B200

# ── Helper ───────────────────────────────────────────────────────────────────
def nearest_64(x: float) -> int:
    return int(round(x / 64)) * 64

def human(n: float) -> str:
    for thresh, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(n) >= thresh:
            return f"{n/thresh:.3f} {suffix}"
    return f"{n:.1f}"

def pct(part: float, whole: float) -> str:
    return f"{100*part/whole:.1f} %"

# ── Configs ──────────────────────────────────────────────────────────────────
CONFIGS = {
    "GPT-2 small":  dict(V=50257, L_ctx=1024, nL=12, d=768,  h=12),
    "GPT-2 medium": dict(V=50257, L_ctx=1024, nL=24, d=1024, h=16),
    "GPT-2 large":  dict(V=50257, L_ctx=1024, nL=36, d=1280, h=20),
    "GPT-2 XL":     dict(V=50257, L_ctx=1024, nL=48, d=1600, h=25),
}

# ── Parameter count ──────────────────────────────────────────────────────────
def count_params(V, d, d_ff, nL):
    emb = V * d
    per_block = 4 * d * d   +   3 * d * d_ff   +   2 * d
    #           QKV+O proj       FFN W1/W2/W3     ln1+ln2
    blocks = per_block * nL
    ln_final = d
    lm_head = d * V
    return {
        "token_embeddings":  emb,
        "MHA (all layers)":  4 * d * d * nL,
        "FFN (all layers)":  3 * d * d_ff * nL,
        "RMSNorm":           (2 * d * nL + d),
        "lm_head":           lm_head,
        "TOTAL":             emb + blocks + ln_final + lm_head,
        "non-embedding":     blocks + ln_final,
    }

# ── FLOPs per (batch=1) forward pass ─────────────────────────────────────────
def flops_fwd(L, d, h, d_ff, nL, V):
    d_k = d // h

    # Per block
    qkv = 4 * 2 * L * d * d                                    # Q/K/V/O projections
    qkt = 2 * L * L * d                                        # QK^T
    av  = 2 * L * L * d                                        # attn·V
    ffn = 3 * 2 * L * d * d_ff                                 # SwiGLU (3 matmuls)
    per_block = qkv + qkt + av + ffn

    lmh = 2 * L * d * V                                        # LM head
    return {
        "QKV projections":       qkv  * nL,
        "QKᵀ (attn scores)":     qkt  * nL,
        "attn · V":              av   * nL,
        "FFN (SwiGLU)":          ffn  * nL,
        "─── all blocks":        per_block * nL,
        "LM head":               lmh,
        "TOTAL forward":         per_block * nL + lmh,
    }

# ── Training time ────────────────────────────────────────────────────────────
def train_sec(total_flops, gpu_tflops, mfu=0.50):
    eff = gpu_tflops * mfu * 1e12
    return total_flops / eff


# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Transformer LM resource accounting")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # (a) GPT-2 XL parameters & memory
    # ══════════════════════════════════════════════════════════════════════════
    print("\n─── (a) GPT-2 XL: Parameters & Memory ───\n")
    c = CONFIGS["GPT-2 XL"]
    d_ff = 4288  # per the reference configuration: nearest_64(8/3 × 1600)
    p = count_params(c["V"], c["d"], d_ff, c["nL"])
    for k, v in p.items():
        print(f"  {k:28s} {v:>16,}   ({human(v)})")
    mem_gb = p["TOTAL"] * 4 / 1e9
    print(f"\n  Memory (float32): {mem_gb:.2f} GB  |  (fp16: {mem_gb/2:.2f} GB)")

    # ══════════════════════════════════════════════════════════════════════════
    # (b) GPT-2 XL FLOPs per forward pass
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n─── (b) GPT-2 XL: FLOPs per forward (L = {c['L_ctx']}) ───\n")
    f = flops_fwd(c["L_ctx"], c["d"], c["h"], d_ff, c["nL"], c["V"])
    T = f["TOTAL forward"]
    for k, v in f.items():
        if k == "─── all blocks":
            print(f"  {'─'*50}")
        print(f"  {k:30s} {v:>20,.0f}  ({human(v)})  {pct(v, T)}")

    # ══════════════════════════════════════════════════════════════════════════
    # (c) Dominant component
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n─── (c) Dominant FLOPs ───\n")
    print("  FFN (SwiGLU) at ~58 % is the largest consumer. The three")
    print("  W1/W2/W3 matmuls cost 3 × 2·L·d·d_ff = 6L·d·d_ff, which is")
    print("  ~6L·d·(8d/3) = 16L·d² vs ~8L·d² for the four QKV projections.")
    print("  Attention scores (QKᵀ + attn·V) are O(L²·d) but at L=1024")
    print("  they're only ~9 %; they dominate only at very long contexts.")

    # ══════════════════════════════════════════════════════════════════════════
    # (d) GPT-2 family
    # ══════════════════════════════════════════════════════════════════════════
    print("\n─── (d) FLOPs across GPT-2 family (L_ctx = 1024) ───\n")
    for name, c_ in CONFIGS.items():
        dff = nearest_64(8 / 3 * c_["d"])
        f_ = flops_fwd(c_["L_ctx"], c_["d"], c_["h"], dff, c_["nL"], c_["V"])
        T_ = f_["TOTAL forward"]
        print(f"  {name:16s}  d={c_['d']:4d}  L={c_['nL']:2d}   h={c_['h']:2d}   d_ff={dff:5d}  "
              f"fwd = {human(T_)}")
        for lbl in ("QKV projections", "QKᵀ (attn scores)", "attn · V", "FFN (SwiGLU)", "LM head"):
            print(f"    {lbl:30s} {pct(f_[lbl], T_)}")
        print()

    print("  Trend as model size grows:")
    print("    · FFN fraction rises from 40 % → 58 % (deeper model → more FFN layers)")
    print("    · LM head drops from 27 % →  5 % (fixed cost amortised over more layers)")
    print("    · QKV rises from 20 % → 29 % (d² term grows with d)")
    print("    · Attention scores (QKᵀ+attn·V) ~12% → ~9%")

    # ══════════════════════════════════════════════════════════════════════════
    # (e) Context-length scaling (GPT-2 XL)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n─── (e) GPT-2 XL: L = 1,024 → 16,384 ───\n")
    f_short = flops_fwd(1024,  c["d"], c["h"], d_ff, c["nL"], c["V"])
    f_long  = flops_fwd(16384, c["d"], c["h"], d_ff, c["nL"], c["V"])
    ratio = f_long["TOTAL forward"] / f_short["TOTAL forward"]
    print(f"  L= 1,024  →  {human(f_short['TOTAL forward'])}")
    print(f"  L=16,384  →  {human(f_long['TOTAL forward'])}  (×{ratio:.0f})")
    print()
    for lbl in ("QKV projections", "QKᵀ (attn scores)", "attn · V", "FFN (SwiGLU)", "LM head"):
        s = pct(f_short[lbl], f_short["TOTAL forward"])
        l = pct(f_long[lbl],  f_long["TOTAL forward"])
        print(f"  {lbl:30s}  {s:>6s}  →  {l:>6s}")
    print(f"\n  Attention (QKᵀ + attn·V) goes from ~9 % → ~62 % because it is O(L²),")
    print(f"  while all other matmuls are O(L). At long context the model becomes")
    print(f"  attention-bound rather than FFN-bound.")

    # =========================================================================
    # Training-time estimates
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training-time estimates (A100, A6000, B200)  —  assume FP16, 50 % MFU")
    print("=" * 70)

    def time_str(sec):
        if sec < 3600:
            return f"{sec/60:.1f} min"
        return f"{sec/3600:.1f} h"

    # ── TinyStories ~17M params  ─────────────────────────────────────────────
    print("\n─── TinyStories   (d=512, 4 layers, 16 heads, L=256) ───\n")
    TS = dict(V=10000, L_ctx=256, nL=4, d=512, h=16, d_ff=1344)
    ts_fwd = flops_fwd(TS["L_ctx"], TS["d"], TS["h"], TS["d_ff"],
                        TS["nL"], TS["V"])["TOTAL forward"]
    ts_tokens = 327_680_000
    ts_batch  = 1024
    ts_steps  = ts_tokens // (ts_batch * TS["L_ctx"])
    ts_total_flops = ts_steps * ts_batch * 3 * ts_fwd       # 3× for fwd+bwd

    # Note: a 17M model can't saturate a big GPU — real MFU is lower.
    # For fair comparison we use a "small-model" effective MFU.
    SMALL_MFU = 0.15                                      # realistic for 17M on B200

    print(f"  Total tokens:        {ts_tokens:,}")
    print(f"  Steps (batch {ts_batch}): {ts_steps}")
    print(f"  Forward / sample:    {human(ts_fwd)}")
    print(f"  Train FLOPs total:   {human(ts_total_flops)}")
    for name, tflops, mfu in [("A100", A100_F16, 0.50), ("A6000", A6000_F16, 0.50),
                                ("B200", B200_F16, SMALL_MFU)]:
        sec = train_sec(ts_total_flops, tflops, mfu)
        print(f"  {name:6s} ({tflops} TFLOPS @ {mfu*100:.0f}% MFU = {tflops*mfu:.0f} T eff): "
              f"  {time_str(sec)}")
    print(f"\n  Note: 17M params is too small to saturate B200 — real MFU ~10-15 %,")
    print(f"  which is why the reference estimate is ~25 min (not the ~3 min pure-compute).")

    # ── GPT-2 XL  hypothetical 400K-step run  ────────────────────────────────
    print(f"\n─── GPT-2 XL, 400K steps, batch=1024, L=1024  (AdamW accounting) ───\n")
    xl_fwd = f["TOTAL forward"]
    xl_steps = 400_000
    xl_batch = 1024
    xl_total = xl_steps * xl_batch * 3 * xl_fwd
    print(f"  Forward / sample:    {human(xl_fwd)}")
    print(f"  Train FLOPs total:   {human(xl_total)}")
    for name, tflops in [("H100", 495.0), ("A100", A100_F16), ("B200", B200_F16)]:
        sec = train_sec(xl_total, tflops, 0.50)
        days = sec / 86400
        print(f"  {name:6s} ({tflops} TFLOPS @ 50% MFU):  {time_str(sec)}  ({days:.0f} days)")
    print(f"\n  This is purely academic — GPT-2 XL at this scale requires a cluster,")
    print(f"  not a single GPU.  H100 at 50% MFU ≈ {time_str(train_sec(xl_total, 495))}.")

    # ── OWT leaderboard: what can you do in 45 min on B200? ─────────────────
    print(f"\n─── OWT leaderboard (45-min B200 budget) ───\n")
    budget_sec = 45 * 60
    # Effective FLOPs available
    budget_flops = budget_sec * B200_F16 * 0.50 * 1e12
    print(f"  FLOPs budget (B200, 50% MFU, 45 min):   {human(budget_flops)}")
    # How many GPT-2 XL tokens could you process?
    xl_per_token = 3 * xl_fwd / 1024              # train FLOPs per token
    xl_tokens = budget_flops / xl_per_token
    print(f"  GPT-2 XL:  that's only {human(xl_tokens)} tokens — far too few to converge.")
    # With a ~100M model:
    mid_d, mid_nL = 768, 12
    mid_dff = nearest_64(8/3*mid_d)               # 2048
    mid_fwd = flops_fwd(1024, mid_d, 12, mid_dff, mid_nL, 50257)["TOTAL forward"]
    mid_per_token = 3 * mid_fwd / 1024
    mid_tokens = budget_flops / mid_per_token
    mid_batch = 512
    mid_steps = mid_tokens / mid_batch
    print(f"\n  100M model (d=768, 12L, L=1024):")
    print(f"    Forward / sample:   {human(mid_fwd)}")
    print(f"    Tokens in budget:   {human(mid_tokens)}")
    print(f"    Steps (batch {mid_batch}):   {mid_steps:.0f}")
    for name, tflops in [("A100", A100_F16), ("A6000", A6000_F16)]:
        sec = budget_flops / (tflops * 0.50 * 1e12)
        print(f"    Same on {name:6s}:   {time_str(sec)}")
    print(f"\n  Leaderboard rule of thumb:")
    print(f"    A100  ≈ 1.4× B200 time   (312 / 450 × same MFU)")
    print(f"    A6000 ≈ 4.1× A100 time   (77.4 / 312)")
    print(f"  So 45 min on B200 → ~65 min on A100 → ~4.4 h on A6000.")
