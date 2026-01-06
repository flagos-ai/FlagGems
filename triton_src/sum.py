import triton
from triton import language as tl


@triton.jit
def sum_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_ids < M

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in tl.range(0, N, BLOCK_N, STAGE):
        col_ids = off + tl.arange(0, BLOCK_N)
        col_mask = col_ids < N
        mask = row_mask[:, None] & col_mask[None, :]

        a = tl.load(in_ptr + row_ids[:, None] * N + col_ids, mask, other=0).to(cdtype)
        acc += a
    out = tl.sum(acc, axis=1)
    tl.store(out_ptr + row_ids, out, row_mask)


if __name__ == "__main__":
    import torch

    m = 1024
    n = 1024 * 16
    x = torch.randn((m, n), device="cuda:0")
    out = torch.empty((m,), device="cuda:0")
    BLOCK_M = 1
    BLOCK_N = 1024
    grid = (triton.cdiv(m, BLOCK_M), 1, 1)
    sum_kernel[grid](x, out, m, n, BLOCK_M, BLOCK_N, STAGE=2, num_warps=4)
    print(out)
    print(x.sum(1))
