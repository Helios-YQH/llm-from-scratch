import math

import einops
import torch


class LinearModule(torch.nn.Module):
    """Linear transformation y = Wx (no bias, following modern LLMs)."""

    def __init__(self, in_features, out_features):
        super(LinearModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return torch.matmul(x, self.weight.t())


class Embedding(torch.nn.Module):
    """Token embedding lookup: maps integer token IDs to dense vectors."""

    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, embedding_dim))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x):
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).  eps=1e-5, float32 upcast."""

    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        return ((x_f32 / rms) * self.weight).to(in_dtype)


class SwiGLU(torch.nn.Module):
    """SwiGLU: W2(SiLU(W1(x)) ⊙ W3(x)).  d_ff ≈ 8/3·d_model, rounded to multiple of 64."""

    def __init__(self, d_model, d_ff):
        super(SwiGLU, self).__init__()
        self.w1 = LinearModule(d_model, d_ff)
        self.w2 = LinearModule(d_ff, d_model)
        self.w3 = LinearModule(d_model, d_ff)

    def forward(self, x):
        return self.w2(torch.sigmoid(self.w1(x)) * self.w1(x) * self.w3(x))


class RoPE(torch.nn.Module):
    """Rotary Position Embeddings (Su et al., 2021).  No learnable parameters."""

    def __init__(self, theta, d_k, max_seq_len):
        super(RoPE, self).__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # θ_i,k = i / Θ^(2(k-1)/d_k)   →  angles (max_seq, d_k/2)
        position = torch.arange(0, max_seq_len).unsqueeze(1)    # (max_seq, 1)
        div_term = torch.exp(
            torch.arange(0, d_k, 2) * -(math.log(float(theta)) / d_k)
        )                                                       # (d_k/2,)
        angles = position * div_term                            # (max_seq, d_k/2)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)

    def forward(self, x, token_positions):
        # x: (…, seq, d_k)    token_positions: (…, seq)
        sin = self.sin[token_positions]          # (…, seq, d_k/2)
        cos = self.cos[token_positions]

        # 若 x 的头部维比 sin 多（attention 中 head 维是多出来的），在 sin/cos 的 batch 后、seq 前插入 1 以广播
        ndim_diff = x.dim() - sin.dim()
        if ndim_diff > 0:
            sin = sin.reshape(sin.shape[:-2] + (1,) * ndim_diff + sin.shape[-2:])
            cos = cos.reshape(cos.shape[:-2] + (1,) * ndim_diff + cos.shape[-2:])

        # 逐对旋转：(x_2k, x_{2k+1}) ← R(cos_k, sin_k)
        out = torch.empty_like(x)
        out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        return out


def softmax(in_features, dim):
    """Numerically stable softmax along `dim`."""
    x_max = in_features.max(dim=dim, keepdim=True).values
    x = torch.exp(in_features - x_max)
    return x / x.sum(dim=dim, keepdim=True)


class CausalMultiHeadAttention(torch.nn.Module):
    """Causal multi-head self-attention with RoPE.

    Q,K,V projections in 3 matrix multiplies (batched across heads via einops.rearrange).
    Causal mask prevents attending to future tokens.
    """

    def __init__(self, d_model, num_heads, max_seq_len=1024, theta=None):
        super(CausalMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = LinearModule(d_model, d_model)
        self.k_proj = LinearModule(d_model, d_model)
        self.v_proj = LinearModule(d_model, d_model)
        self.output_proj = LinearModule(d_model, d_model)

        if theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x, token_positions=None):
        batch, seq_len, _ = x.shape

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        Q = einops.rearrange(self.q_proj(x),
            "batch seq (head dim) -> batch head seq dim", head=self.num_heads)
        K = einops.rearrange(self.k_proj(x),
            "batch seq (head dim) -> batch head seq dim", head=self.num_heads)
        V = einops.rearrange(self.v_proj(x),
            "batch seq (head dim) -> batch head seq dim", head=self.num_heads)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        scale = self.d_k ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        scores = scores.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float("-inf"))
        attn = softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = einops.rearrange(context,
            "batch head seq dim -> batch seq (head dim)")
        return self.output_proj(context)


class FeedForward(torch.nn.Module):
    """Position-wise feed-forward (SwiGLU).  Flat module so state-dict keys are ffn.w1/w2/w3."""

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.w1 = LinearModule(d_model, d_ff)
        self.w2 = LinearModule(d_ff, d_model)
        self.w3 = LinearModule(d_model, d_ff)

    def forward(self, x):
        return self.w2(torch.sigmoid(self.w1(x)) * self.w1(x) * self.w3(x))


class TransformerBlock(torch.nn.Module):
    """Pre-norm Transformer block:  y = x + MHA(RMSNorm(x)),  z = y + FFN(RMSNorm(y))."""

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, token_positions=None):
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device)
        x = self.attn(x, token_positions) + x
        x = self.ffn(x) + x
        return x


class TransformerLM(torch.nn.Module):
    """Full Transformer LM.

    token IDs → Embedding → num_layers × TransformerBlock → final RMSNorm → LM head → logits.
    """

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = LinearModule(d_model, vocab_size)

    def forward(self, x, token_positions=None):
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device)
        h = self.token_embeddings(x)
        for layer in self.layers:
            h = layer(h, token_positions)
        h = self.ln_final(h)
        return self.lm_head(h)
