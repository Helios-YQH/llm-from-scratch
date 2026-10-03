"""§4.2.2 FlashAttention-2 forward pass.

(a) Pure PyTorch implementation with online softmax (no Triton).
This is the reference used to debug the Triton kernel; it is slow on purpose.

Interface (matches tests/adapters.py + tests/test_attention.py):
    class FlashAttentionPytorch(torch.autograd.Function):
        forward(ctx, Q, K, V, is_causal=False) -> O
        backward(ctx, dO) -> dQ, dK, dV

Inputs are (B, S, D) tensors (batch x heads already flattened). The forward
must save a tensor of shape (B, S) equal to logsumexp of attention scores, so
the test can extract it from saved_tensors.
"""
from __future__ import annotations

import math

import torch

from lm_basics.nn_utils import softmax

try:
    import triton
    import triton.language as tl
    import triton.testing
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def attention_reference(Q, K, V, is_causal=False):
    """Standard (non-flash) attention used to cross-check. Returns O and L."""
    d = Q.shape[-1]
    S = torch.einsum("bqd,bkd->bqk", Q, K) * (1 / math.sqrt(d))
    if is_causal:
        nq, nk = Q.shape[-2], K.shape[-2]
        S = torch.where(
            torch.arange(nq, device=S.device)[None, :, None] >= torch.arange(nk, device=S.device)[None, None, :],
            S,
            -1e6,
        )
    P = softmax(S, dim=-1)
    O = torch.einsum("bqk,bkd->bqd", P, V)
    L = torch.logsumexp(S, dim=-1)
    return O, L


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """FlashAttention-2 forward via online softmax (tiled, pure PyTorch).

        Args:
            Q, K, V: (B, S, D) tensors (heads flattened into batch dim).
            is_causal: bool, apply causal masking.

        Returns:
            O: (B, S, D) output.
        """
        B, S, D = Q.shape
        scale = 1 / math.sqrt(D)

        # Tile sizes (must be >= 16 per PDF; S and D are powers of 2 in tests).
        Q_TILE = 32 if S >= 32 else S
        K_TILE = 32 if S >= 32 else S

        # Running statistics per query tile.
        # We accumulate output row-wise: O[b, q_tile, :], m[b, q_tile], l[b, q_tile]
        O = torch.zeros(B, S, D, dtype=Q.dtype, device=Q.device)
        m = torch.full((B, S), float("-inf"), dtype=torch.float32, device=Q.device)
        l = torch.zeros(B, S, dtype=torch.float32, device=Q.device)

        # To save L for backward we need it per-row (B, S) — computed as we go.

        for q_start in range(0, S, Q_TILE):
            q_end = min(q_start + Q_TILE, S)
            Q_tile = Q[:, q_start:q_end, :]  # (B, Q_TILE, D)

            # running max/sum for this query tile, reset per tile
            m_tile = torch.full((B, q_end - q_start), float("-inf"), dtype=torch.float32, device=Q.device)
            l_tile = torch.zeros(B, q_end - q_start, dtype=torch.float32, device=Q.device)
            acc_tile = torch.zeros(B, q_end - q_start, D, dtype=Q.dtype, device=Q.device)

            for k_start in range(0, S, K_TILE):
                k_end = min(k_start + K_TILE, S)
                K_tile = K[:, k_start:k_end, :]  # (B, K_TILE, D)
                V_tile = V[:, k_start:k_end, :]  # (B, K_TILE, D)

                # attention score tile (B, Q_TILE, K_TILE)
                S_tile = torch.einsum("bqd,bkd->bqk", Q_tile, K_tile) * scale

                if is_causal:
                    # causal mask within this tile: query index >= key index
                    qi = torch.arange(q_start, q_end, device=S_tile.device)[None, :, None]
                    ki = torch.arange(k_start, k_end, device=S_tile.device)[None, None, :]
                    mask = qi >= ki  # (1, Q_TILE, K_TILE)
                    S_tile = torch.where(mask, S_tile, -1e6)

                # online softmax update
                m_new = torch.maximum(m_tile, S_tile.max(dim=-1).values)  # (B, Q_TILE)
                alpha = torch.exp(m_tile - m_new)  # (B, Q_TILE)
                P_tile = torch.exp(S_tile - m_new.unsqueeze(-1))  # (B, Q_TILE, K_TILE)
                l_tile = l_tile * alpha + P_tile.sum(dim=-1)  # (B, Q_TILE)
                acc_tile = acc_tile * alpha.unsqueeze(-1) + torch.einsum("bqk,bkd->bqd", P_tile, V_tile)
                m_tile = m_new

            # write finished query tile
            O[:, q_start:q_end, :] = acc_tile / l_tile.unsqueeze(-1)
            m[:, q_start:q_end] = m_tile
            l[:, q_start:q_end] = l_tile

        # logsumexp = m + log(l)
        L = m + torch.log(l)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None


def get_flashattention_autograd_function_pytorch():
    return FlashAttentionPytorch


# ---------------------------------------------------------------------------
# FlashAttention backward (PDF Eq 13-19), plain PyTorch, torch.compile'd
# ---------------------------------------------------------------------------

def _flash_backward_core(Q, K, V, O, L, dO, is_causal, scale):
    """Compute dQ, dK, dV via recomputation (Eq 13-19). Torch-compilable."""
    B, S, D = Q.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # Eq 17: D = rowsum(dO * O)
    D = (dO * O).sum(dim=-1)  # (B, S)

    # We recompute P = exp(S - L) per tile (no need to store S^2).
    # Simple non-tiled implementation for clarity (torch.compile will fuse).
    # Recompute full S (this is the "recomputation" part; memory-wise it's
    # still O(S^2) here but torch.compile can be given tiles; PDF only
    # requires correctness for the torch.compile implementation).
    S_mat = torch.einsum("bqd,bkd->bqk", Q, K) * scale
    if is_causal:
        nq, nk = S_mat.shape[-2], S_mat.shape[-1]
        S_mat = torch.where(
            torch.arange(nq, device=S_mat.device)[None, :, None] >= torch.arange(nk, device=S_mat.device)[None, None, :],
            S_mat,
            -1e6,
        )
    P = torch.exp(S_mat - L.unsqueeze(-1))  # (B, S, S) softmax probs

    # Eq 15: dV = P^T @ dO
    dV = torch.einsum("bqk,bqd->bkd", P, dO)
    # Eq 16: dP = dO @ V^T
    dP = torch.einsum("bqd,bkd->bqk", dO, V)
    # Eq 17 (inside): dS = P * (dP - D)
    dS = P * (dP - D.unsqueeze(-1))
    # Eq 18: dQ = dS @ K * scale
    dQ = torch.einsum("bqk,bkd->bqd", dS, K) * scale
    # Eq 19: dK = dS^T @ Q * scale
    dK = torch.einsum("bqk,bqd->bkd", dS, Q) * scale

    return dQ, dK, dV


def flash_backward_compiled(Q, K, V, O, L, dO, is_causal):
    """torch.compile'd backward. Used by both Function classes.

    Falls back to the plain function on CPU, where torch.compile may not be
    available (e.g. Windows without a C++ compiler).
    """
    D = Q.shape[-1]
    scale = 1.0 / (D ** 0.5)
    if Q.is_cuda:
        fn = torch.compile(_flash_backward_core, fullgraph=False)
        return fn(Q, K, V, O, L, dO, is_causal, scale)
    return _flash_backward_core(Q, K, V, O, L, dO, is_causal, scale)


# ---------------------------------------------------------------------------
# (b) Triton kernel for the FlashAttention-2 forward pass (Algorithm 1)
# ---------------------------------------------------------------------------

def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices: one program per (query_tile, batch)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Block pointer setup (offset each tensor by batch index)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load Q tile once (stays in SRAM across the key loop)
    q_tile = tl.load(Q_block_ptr)  # (Q_TILE, D)
    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # (Q_TILE,)

    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for key_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_tile = tl.load(K_block_ptr)  # (K_TILE, D)
        v_tile = tl.load(V_block_ptr)  # (K_TILE, D)

        # attention scores: (Q_TILE, K_TILE) = Q @ K^T * scale
        s = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * scale

        if is_causal:
            # causal mask: keep query_idx >= key_idx. key_tile spans
            # [key_tile_index*K_TILE, +K_TILE)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (K_TILE,)
            causal_mask = q_idx[:, None] >= k_idx[None, :]  # (Q_TILE, K_TILE)
            s = tl.where(causal_mask, s, float("-inf"))

        # online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile, out_dtype=tl.float32)
        m_i = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o_tile = acc / l_i[:, None]
    tl.store(O_block_ptr, o_tile.to(O_block_ptr.type.element_ty))
    L_tile = m_i + tl.log(l_i)
    tl.store(L_block_ptr, L_tile)


if _HAS_TRITON:
    flash_fwd_kernel = triton.jit(flash_fwd_kernel)


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 forward using the Triton kernel."""

    Q_TILE_SIZE = 64
    K_TILE_SIZE = 64

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q, K, V: (B, S, D) with B = batch*heads (already flattened).
        B, S, D = Q.shape
        scale = 1.0 / (D ** 0.5)
        assert S % FlashAttentionTriton.Q_TILE_SIZE == 0 or S < FlashAttentionTriton.Q_TILE_SIZE

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        O = torch.empty_like(Q)
        L = torch.empty((B, S), dtype=torch.float32, device=Q.device)

        grid = (triton.cdiv(S, FlashAttentionTriton.Q_TILE_SIZE), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=S, N_KEYS=S,
            scale=scale,
            D=D,
            Q_TILE_SIZE=FlashAttentionTriton.Q_TILE_SIZE,
            K_TILE_SIZE=FlashAttentionTriton.K_TILE_SIZE,
            is_causal=is_causal,
            num_stages=1,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None


def get_flashattention_autograd_function_triton():
    return FlashAttentionTriton


# ---------------------------------------------------------------------------
# (4.2.3, optional) Tiled Triton backward pass (Algorithm 2).
#
# Key trick: compute S twice — pass 1 goes over KEY tiles (each program sums
# dK/dV contributions across ALL query tiles, no atomics needed), pass 2 goes
# over QUERY tiles (each program sums dQ contributions across all key tiles).
# ---------------------------------------------------------------------------

def flash_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """Pass 1: one program per (key_tile, batch). Computes full dK_j and dV_j."""
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    k_idx_abs = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (K_TILE,)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D), strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D), strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )

    k_tile = tl.load(K_block_ptr)
    v_tile = tl.load(V_block_ptr)

    dK_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for query_tile_index in range(0, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_idx_abs = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,), strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db,
            shape=(N_QUERIES,), strides=(stride_dq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,), order=(0,),
        )

        q_tile = tl.load(Q_block_ptr)
        dO_tile = tl.load(dO_block_ptr)
        L_tile = tl.load(L_block_ptr)  # (Q_TILE,)
        D_tile = tl.load(D_block_ptr)  # (Q_TILE,)

        s = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * scale  # (Q_TILE, K_TILE)
        if is_causal:
            mask = q_idx_abs[:, None] >= k_idx_abs[None, :]
            s = tl.where(mask, s, float("-inf"))
        p = tl.exp(s - L_tile[:, None])  # (Q_TILE, K_TILE)

        dV_acc += tl.dot(tl.trans(p.to(dO_tile.dtype)), dO_tile, out_dtype=tl.float32)  # (K_TILE, D)

        # dP = P ⊙ (dO @ V^T - D)
        dp = p * (tl.dot(dO_tile, tl.trans(v_tile), out_dtype=tl.float32) - D_tile[:, None])
        dK_acc += tl.dot(tl.trans(dp.to(q_tile.dtype)), q_tile, out_dtype=tl.float32) * scale

    tl.store(dK_block_ptr, dK_acc.to(dK_block_ptr.type.element_ty))
    tl.store(dV_block_ptr, dV_acc.to(dV_block_ptr.type.element_ty))


def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """Pass 2: one program per (query_tile, batch). Computes full dQ_i."""
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_idx_abs = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # (Q_TILE,)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,), order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,), strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,), order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D), strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )

    q_tile = tl.load(Q_block_ptr)
    dO_tile = tl.load(dO_block_ptr)
    L_tile = tl.load(L_block_ptr)
    D_tile = tl.load(D_block_ptr)

    dQ_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for key_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_idx_abs = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )

        k_tile = tl.load(K_block_ptr)
        v_tile = tl.load(V_block_ptr)

        s = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * scale
        if is_causal:
            mask = q_idx_abs[:, None] >= k_idx_abs[None, :]
            s = tl.where(mask, s, float("-inf"))
        p = tl.exp(s - L_tile[:, None])

        dp = p * (tl.dot(dO_tile, tl.trans(v_tile), out_dtype=tl.float32) - D_tile[:, None])
        dQ_acc += tl.dot(dp.to(k_tile.dtype), k_tile, out_dtype=tl.float32) * scale

    tl.store(dQ_block_ptr, dQ_acc.to(dQ_block_ptr.type.element_ty))


class FlashAttentionTritonBwd(torch.autograd.Function):
    """FlashAttention-2 with Triton forward AND tiled Triton backward (4.2.3)."""

    Q_TILE_SIZE = 64
    K_TILE_SIZE = 64

    @staticmethod
    def _tile_sizes(D: int):
        # Larger D needs more shared memory per tile (A6000 limit ~99KB);
        # shrink tiles for D > 64 so the kernel fits (PDF: adjust tile sizes).
        if D > 64:
            return 32, 32
        return FlashAttentionTritonBwd.Q_TILE_SIZE, FlashAttentionTritonBwd.K_TILE_SIZE

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, S, D = Q.shape
        scale = 1.0 / (D ** 0.5)
        Q_TILE, K_TILE = FlashAttentionTritonBwd._tile_sizes(D)
        Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
        O = torch.empty_like(Q)
        L = torch.empty((B, S), dtype=torch.float32, device=Q.device)
        grid = (triton.cdiv(S, Q_TILE), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=S, N_KEYS=S,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE,
            K_TILE_SIZE=K_TILE,
            is_causal=is_causal,
            num_stages=1,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, S, D = Q.shape
        scale = 1.0 / (D ** 0.5)
        Q_TILE, K_TILE = FlashAttentionTritonBwd._tile_sizes(D)
        dO = dO.contiguous()

        # D = rowsum(dO * O)  (Eq 17, precomputed in global memory)
        D_vec = (dO * O).sum(dim=-1, dtype=torch.float32).contiguous()  # (B, S)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        # Pass 1: dK, dV (one program per key tile)
        grid1 = (triton.cdiv(S, K_TILE), B)
        flash_bwd_dk_dv_kernel[grid1](
            Q, K, V, dO, L, D_vec, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES=S, N_KEYS=S,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE,
            K_TILE_SIZE=K_TILE,
            is_causal=ctx.is_causal,
            num_stages=1,
        )

        # Pass 2: dQ (one program per query tile)
        grid2 = (triton.cdiv(S, Q_TILE), B)
        flash_bwd_dq_kernel[grid2](
            Q, K, V, dO, L, D_vec, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_QUERIES=S, N_KEYS=S,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE,
            K_TILE_SIZE=K_TILE,
            is_causal=ctx.is_causal,
            num_stages=1,
        )

        return dQ, dK, dV, None


if _HAS_TRITON:
    flash_bwd_dk_dv_kernel = triton.jit(flash_bwd_dk_dv_kernel)
    flash_bwd_dq_kernel = triton.jit(flash_bwd_dq_kernel)
