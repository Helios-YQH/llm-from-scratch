"""§4.2.1 weighted_sum Triton example + PyTorch/CUDA/Triton performance comparison.

Implements `(weight * x).sum(-1)` three ways:
  1. Pure PyTorch (reference)
  2. Triton kernel wrapped in an autograd.Function (forward + backward)
  3. torch.compile'd PyTorch (as a "CUDA/compile" middle ground)

Compares correctness (allclose against PyTorch) and speed (triton.testing.do_bench).

Run on the GPU server (needs triton). Not runnable on local CPU-only torch.
"""
import torch
import triton
import triton.language as tl
import triton.testing


# ---------------- Triton kernels ----------------

@triton.jit
def weighted_sum_fwd_kernel(
    x_ptr, weight_ptr, output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(row * weight[None, :], axis=1)
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_bwd_kernel(
    x_ptr, weight_ptr, grad_output_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,), order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE,), order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE), order=(1, 0),
    )

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        # grad_x = grad_output * weight (outer product)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
        # partial grad_weight = x * grad_output summed over rows
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D = x.shape[-1]
        input_shape = x.shape
        x2d = x.reshape(-1, D)
        assert x2d.is_contiguous(), "need contiguous x"
        ctx.save_for_backward(x2d, weight)
        ctx.ROWS_TILE_SIZE = 16
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 4
        output_dims = x.shape[:-1]
        y = torch.empty(output_dims, device=x.device, dtype=torch.float32)
        n_rows = y.numel()
        weighted_sum_fwd_kernel[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x2d, weight, y,
            x2d.stride(0), x2d.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape
        grad_out2d = grad_out.reshape(-1)
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)
        weighted_sum_bwd_kernel[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight, grad_out2d,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out2d.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x.reshape(grad_out.shape + (D,)) if grad_out.dim() > 1 else grad_x, grad_weight


# ---------------- reference implementations ----------------

def weighted_sum_pytorch(x, weight):
    return (weight * x).sum(-1)


def main():
    torch.manual_seed(0)
    B, D = 4096, 1024
    x = torch.randn(B, D, device="cuda", requires_grad=True)
    weight = torch.randn(D, device="cuda", requires_grad=True)

    # correctness: forward
    ref_fwd = weighted_sum_pytorch(x, weight)
    trit_fwd = WeightedSumFunc.apply(x, weight)
    ok_fwd = torch.allclose(ref_fwd, trit_fwd, atol=1e-3)
    print(f"forward correctness (allclose): {ok_fwd}")

    # correctness: backward (use the SAME scalar loss for both)
    loss_ref = (ref_fwd * 2).sum()
    loss_ref.backward(retain_graph=True)
    gx_ref, gw_ref = x.grad.clone(), weight.grad.clone()
    x.grad = None; weight.grad = None

    loss_trit = (trit_fwd * 2).sum()
    loss_trit.backward()
    gx_trit, gw_trit = x.grad.clone(), weight.grad.clone()
    ok_gx = torch.allclose(gx_ref, gx_trit, atol=1e-3)
    ok_gw = torch.allclose(gw_ref, gw_trit, atol=1e-3)
    print(f"backward dL/dx correctness: {ok_gx}, dL/dw correctness: {ok_gw}")

    # torch.compile reference
    compiled = torch.compile(weighted_sum_pytorch)

    # performance comparison
    x_bench = torch.randn(B, D, device="cuda", requires_grad=True)
    w_bench = torch.randn(D, device="cuda", requires_grad=True)

    def bench_fwd(fn):
        return triton.testing.do_bench(lambda: fn(x_bench, w_bench))

    def bench_full(fn):
        def full():
            out = fn(x_bench, w_bench)
            out.sum().backward()
        return triton.testing.do_bench(full)

    t_pytorch_f = bench_fwd(weighted_sum_pytorch)
    t_pytorch_full = bench_full(weighted_sum_pytorch)
    t_compile_f = bench_fwd(compiled)
    t_compile_full = bench_full(compiled)
    t_triton_f = bench_fwd(WeightedSumFunc.apply)
    t_triton_full = bench_full(WeightedSumFunc.apply)

    print(f"\n{'impl':<12} {'fwd (ms)':>10} {'full fwd+bwd (ms)':>18}")
    print("-" * 46)
    print(f"{'PyTorch':<12} {t_pytorch_f:>10.3f} {t_pytorch_full:>18.3f}")
    print(f"{'torch.compile':<12} {t_compile_f:>10.3f} {t_compile_full:>18.3f}")
    print(f"{'Triton':<12} {t_triton_f:>10.3f} {t_triton_full:>18.3f}")
    print(f"\nTriton speedup vs PyTorch: fwd {t_pytorch_f/t_triton_f:.1f}x, full {t_pytorch_full/t_triton_full:.1f}x")


if __name__ == "__main__":
    main()
