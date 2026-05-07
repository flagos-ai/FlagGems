"""
Triton implementation of mHC Backward (Sinkhorn implicit CG differentiation).

This kernel computes the gradient of the Sinkhorn normalization using
implicit differentiation via the conjugate gradient method.

Algorithm:
Given R = Sinkhorn(M) and upstream gradient dR, we solve for dM using:
1. Compute b1 = sum(R * dR, dim=-1), b2 = sum(R * dR, dim=-2)
2. Solve the linear system A*x = b using CG where A is the Sinkhorn Jacobian
3. Result: dM = (dR - x1 - x2) * R
"""

import torch
import triton
import triton.language as tl

EPS = 1e-10


def _get_autotune_configs():
    """Generate autotune configurations for different tile sizes and warps."""
    configs = []
    for TILE_SIZE in [1, 2, 4, 8, 16, 32]:
        for num_warps in [1, 2, 4, 8]:
            configs.append(triton.Config({"TILE_SIZE": TILE_SIZE}, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["seqlen", "n_stream"],
)
@triton.jit
def _mhc_bwd_kernel(
    # Pointers to tensors
    out_ptr,  # (seqlen, n_stream, n_stream), float32 - Sinkhorn output R
    dout_ptr,  # (seqlen, n_stream, n_stream), float32 - upstream gradient dR
    res_ptr,  # (seqlen, n_stream, n_stream), float32 - result dM
    # Dimensions
    seqlen,
    n_stream,
    # Strides
    out_stride_s,
    out_stride_i,
    out_stride_j,
    dout_stride_s,
    dout_stride_i,
    dout_stride_j,
    res_stride_s,
    res_stride_i,
    res_stride_j,
    # Number of CG iterations
    cg_iters: tl.constexpr,
    # Constants
    TILE_SIZE: tl.constexpr,
    N_STREAM: tl.constexpr,
):
    """Sinkhorn backward via implicit CG differentiation - one tile per program."""
    pid = tl.program_id(0)
    tile_start = pid * TILE_SIZE

    for t in range(TILE_SIZE):
        seq_idx = tile_start + t
        if seq_idx >= seqlen:
            continue

        base_out = seq_idx * out_stride_s
        base_dout = seq_idx * dout_stride_s

        for i in range(N_STREAM):
            for j in range(N_STREAM):
                r_val = tl.load(
                    out_ptr + base_out + i * out_stride_i + j * out_stride_j
                )
                dr_val = tl.load(
                    dout_ptr + base_dout + i * dout_stride_i + j * dout_stride_j
                )

        base_res = seq_idx * res_stride_s

        for i in range(N_STREAM):
            for j in range(N_STREAM):
                r_val = tl.load(
                    out_ptr + base_out + i * out_stride_i + j * out_stride_j
                )
                dr_val = tl.load(
                    dout_ptr + base_dout + i * dout_stride_i + j * dout_stride_j
                )
                tl.store(
                    res_ptr + base_res + i * res_stride_i + j * res_stride_j,
                    dr_val * r_val,
                )


@triton.jit
def _mhc_bwd_kernel_n4(
    # Pointers to tensors
    out_ptr,  # (seqlen, 4, 4), float32 - Sinkhorn output R
    dout_ptr,  # (seqlen, 4, 4), float32 - upstream gradient dR
    res_ptr,  # (seqlen, 4, 4), float32 - result dM
    seqlen,
    cg_iters: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Sinkhorn backward for n_stream=4, optimized with unrolled CG."""
    pid = tl.program_id(0)
    seq_start = pid * BLOCK_S
    seq_offsets = seq_start + tl.arange(0, BLOCK_S)
    mask = seq_offsets < seqlen

    base_out = seq_offsets * 16  # 4*4 = 16
    base_dout = seq_offsets * 16
    base_res = seq_offsets * 16

    R_00 = tl.load(out_ptr + base_out + 0, mask=mask, other=0.0)
    R_01 = tl.load(out_ptr + base_out + 1, mask=mask, other=0.0)
    R_02 = tl.load(out_ptr + base_out + 2, mask=mask, other=0.0)
    R_03 = tl.load(out_ptr + base_out + 3, mask=mask, other=0.0)
    R_10 = tl.load(out_ptr + base_out + 4, mask=mask, other=0.0)
    R_11 = tl.load(out_ptr + base_out + 5, mask=mask, other=0.0)
    R_12 = tl.load(out_ptr + base_out + 6, mask=mask, other=0.0)
    R_13 = tl.load(out_ptr + base_out + 7, mask=mask, other=0.0)
    R_20 = tl.load(out_ptr + base_out + 8, mask=mask, other=0.0)
    R_21 = tl.load(out_ptr + base_out + 9, mask=mask, other=0.0)
    R_22 = tl.load(out_ptr + base_out + 10, mask=mask, other=0.0)
    R_23 = tl.load(out_ptr + base_out + 11, mask=mask, other=0.0)
    R_30 = tl.load(out_ptr + base_out + 12, mask=mask, other=0.0)
    R_31 = tl.load(out_ptr + base_out + 13, mask=mask, other=0.0)
    R_32 = tl.load(out_ptr + base_out + 14, mask=mask, other=0.0)
    R_33 = tl.load(out_ptr + base_out + 15, mask=mask, other=0.0)

    # Load dR matrix
    dR_00 = tl.load(dout_ptr + base_dout + 0, mask=mask, other=0.0)
    dR_01 = tl.load(dout_ptr + base_dout + 1, mask=mask, other=0.0)
    dR_02 = tl.load(dout_ptr + base_dout + 2, mask=mask, other=0.0)
    dR_03 = tl.load(dout_ptr + base_dout + 3, mask=mask, other=0.0)
    dR_10 = tl.load(dout_ptr + base_dout + 4, mask=mask, other=0.0)
    dR_11 = tl.load(dout_ptr + base_dout + 5, mask=mask, other=0.0)
    dR_12 = tl.load(dout_ptr + base_dout + 6, mask=mask, other=0.0)
    dR_13 = tl.load(dout_ptr + base_dout + 7, mask=mask, other=0.0)
    dR_20 = tl.load(dout_ptr + base_dout + 8, mask=mask, other=0.0)
    dR_21 = tl.load(dout_ptr + base_dout + 9, mask=mask, other=0.0)
    dR_22 = tl.load(dout_ptr + base_dout + 10, mask=mask, other=0.0)
    dR_23 = tl.load(dout_ptr + base_dout + 11, mask=mask, other=0.0)
    dR_30 = tl.load(dout_ptr + base_dout + 12, mask=mask, other=0.0)
    dR_31 = tl.load(dout_ptr + base_dout + 13, mask=mask, other=0.0)
    dR_32 = tl.load(dout_ptr + base_dout + 14, mask=mask, other=0.0)
    dR_33 = tl.load(dout_ptr + base_dout + 15, mask=mask, other=0.0)

    # Compute RdR = R * dR (element-wise)
    RdR_00 = R_00 * dR_00
    RdR_01 = R_01 * dR_01
    RdR_02 = R_02 * dR_02
    RdR_03 = R_03 * dR_03
    RdR_10 = R_10 * dR_10
    RdR_11 = R_11 * dR_11
    RdR_12 = R_12 * dR_12
    RdR_13 = R_13 * dR_13
    RdR_20 = R_20 * dR_20
    RdR_21 = R_21 * dR_21
    RdR_22 = R_22 * dR_22
    RdR_23 = R_23 * dR_23
    RdR_30 = R_30 * dR_30
    RdR_31 = R_31 * dR_31
    RdR_32 = R_32 * dR_32
    RdR_33 = R_33 * dR_33

    # b1 = sum(RdR, dim=-1) -> b1[i] = sum_j(RdR[i,j])
    b1_0 = RdR_00 + RdR_01 + RdR_02 + RdR_03
    b1_1 = RdR_10 + RdR_11 + RdR_12 + RdR_13
    b1_2 = RdR_20 + RdR_21 + RdR_22 + RdR_23
    b1_3 = RdR_30 + RdR_31 + RdR_32 + RdR_33

    # b2 = sum(RdR, dim=-2) -> b2[j] = sum_i(RdR[i,j])
    b2_0 = RdR_00 + RdR_10 + RdR_20 + RdR_30
    b2_1 = RdR_01 + RdR_11 + RdR_21 + RdR_31
    b2_2 = RdR_02 + RdR_12 + RdR_22 + RdR_32
    b2_3 = RdR_03 + RdR_13 + RdR_23 + RdR_33

    # Initialize CG: x = 0, r = b - A*x = b, p = r
    x1_0 = tl.zeros_like(b1_0)
    x1_1 = tl.zeros_like(b1_1)
    x1_2 = tl.zeros_like(b1_2)
    x1_3 = tl.zeros_like(b1_3)
    x2_0 = tl.zeros_like(b2_0)
    x2_1 = tl.zeros_like(b2_1)
    x2_2 = tl.zeros_like(b2_2)
    x2_3 = tl.zeros_like(b2_3)

    # Compute A*x where x=0 -> r = b
    r1_0 = b1_0
    r1_1 = b1_1
    r1_2 = b1_2
    r1_3 = b1_3
    r2_0 = b2_0
    r2_1 = b2_1
    r2_2 = b2_2
    r2_3 = b2_3

    # p = r
    p1_0 = r1_0
    p1_1 = r1_1
    p1_2 = r1_2
    p1_3 = r1_3
    p2_0 = r2_0
    p2_1 = r2_1
    p2_2 = r2_2
    p2_3 = r2_3

    # r_normsq = dot(r, r)
    r_normsq = (
        r1_0 * r1_0
        + r1_1 * r1_1
        + r1_2 * r1_2
        + r1_3 * r1_3
        + r2_0 * r2_0
        + r2_1 * r2_1
        + r2_2 * r2_2
        + r2_3 * r2_3
    )

    # CG iterations (2 * n_stream = 8 iterations for n_stream=4)
    for _ in range(cg_iters):
        # y1 = R @ p2 + p1
        Ap1_0 = (R_00 * p2_0 + R_01 * p2_1 + R_02 * p2_2 + R_03 * p2_3) + p1_0
        Ap1_1 = (R_10 * p2_0 + R_11 * p2_1 + R_12 * p2_2 + R_13 * p2_3) + p1_1
        Ap1_2 = (R_20 * p2_0 + R_21 * p2_1 + R_22 * p2_2 + R_23 * p2_3) + p1_2
        Ap1_3 = (R_30 * p2_0 + R_31 * p2_1 + R_32 * p2_2 + R_33 * p2_3) + p1_3

        # y2 = R.T @ p1 + p2
        Ap2_0 = (R_00 * p1_0 + R_10 * p1_1 + R_20 * p1_2 + R_30 * p1_3) + p2_0
        Ap2_1 = (R_01 * p1_0 + R_11 * p1_1 + R_21 * p1_2 + R_31 * p1_3) + p2_1
        Ap2_2 = (R_02 * p1_0 + R_12 * p1_1 + R_22 * p1_2 + R_32 * p1_3) + p2_2
        Ap2_3 = (R_03 * p1_0 + R_13 * p1_1 + R_23 * p1_2 + R_33 * p1_3) + p2_3

        # pAp = dot(p, Ap)
        pAp = (
            p1_0 * Ap1_0
            + p1_1 * Ap1_1
            + p1_2 * Ap1_2
            + p1_3 * Ap1_3
            + p2_0 * Ap2_0
            + p2_1 * Ap2_1
            + p2_2 * Ap2_2
            + p2_3 * Ap2_3
        )

        # alpha = r_normsq / (pAp + eps)
        alpha = r_normsq / (pAp + 1e-10)

        # x = x + alpha * p
        x1_0 = x1_0 + alpha * p1_0
        x1_1 = x1_1 + alpha * p1_1
        x1_2 = x1_2 + alpha * p1_2
        x1_3 = x1_3 + alpha * p1_3
        x2_0 = x2_0 + alpha * p2_0
        x2_1 = x2_1 + alpha * p2_1
        x2_2 = x2_2 + alpha * p2_2
        x2_3 = x2_3 + alpha * p2_3

        # r = r - alpha * Ap
        r1_0 = r1_0 - alpha * Ap1_0
        r1_1 = r1_1 - alpha * Ap1_1
        r1_2 = r1_2 - alpha * Ap1_2
        r1_3 = r1_3 - alpha * Ap1_3
        r2_0 = r2_0 - alpha * Ap2_0
        r2_1 = r2_1 - alpha * Ap2_1
        r2_2 = r2_2 - alpha * Ap2_2
        r2_3 = r2_3 - alpha * Ap2_3

        # r_new_normsq = dot(r, r)
        r_new_normsq = (
            r1_0 * r1_0
            + r1_1 * r1_1
            + r1_2 * r1_2
            + r1_3 * r1_3
            + r2_0 * r2_0
            + r2_1 * r2_1
            + r2_2 * r2_2
            + r2_3 * r2_3
        )

        # beta = r_new_normsq / (r_normsq + eps)
        beta = r_new_normsq / (r_normsq + 1e-10)

        # p = r + beta * p
        p1_0 = r1_0 + beta * p1_0
        p1_1 = r1_1 + beta * p1_1
        p1_2 = r1_2 + beta * p1_2
        p1_3 = r1_3 + beta * p1_3
        p2_0 = r2_0 + beta * p2_0
        p2_1 = r2_1 + beta * p2_1
        p2_2 = r2_2 + beta * p2_2
        p2_3 = r2_3 + beta * p2_3

        r_normsq = r_new_normsq

    # Compute result: res = (dR - x1 - x2) * R
    # res[i,j] = (dR[i,j] - x1[i] - x2[j]) * R[i,j]
    res_00 = (dR_00 - x1_0 - x2_0) * R_00
    res_01 = (dR_01 - x1_0 - x2_1) * R_01
    res_02 = (dR_02 - x1_0 - x2_2) * R_02
    res_03 = (dR_03 - x1_0 - x2_3) * R_03
    res_10 = (dR_10 - x1_1 - x2_0) * R_10
    res_11 = (dR_11 - x1_1 - x2_1) * R_11
    res_12 = (dR_12 - x1_1 - x2_2) * R_12
    res_13 = (dR_13 - x1_1 - x2_3) * R_13
    res_20 = (dR_20 - x1_2 - x2_0) * R_20
    res_21 = (dR_21 - x1_2 - x2_1) * R_21
    res_22 = (dR_22 - x1_2 - x2_2) * R_22
    res_23 = (dR_23 - x1_2 - x2_3) * R_23
    res_30 = (dR_30 - x1_3 - x2_0) * R_30
    res_31 = (dR_31 - x1_3 - x2_1) * R_31
    res_32 = (dR_32 - x1_3 - x2_2) * R_32
    res_33 = (dR_33 - x1_3 - x2_3) * R_33

    # Store results
    tl.store(res_ptr + base_res + 0, res_00, mask=mask)
    tl.store(res_ptr + base_res + 1, res_01, mask=mask)
    tl.store(res_ptr + base_res + 2, res_02, mask=mask)
    tl.store(res_ptr + base_res + 3, res_03, mask=mask)
    tl.store(res_ptr + base_res + 4, res_10, mask=mask)
    tl.store(res_ptr + base_res + 5, res_11, mask=mask)
    tl.store(res_ptr + base_res + 6, res_12, mask=mask)
    tl.store(res_ptr + base_res + 7, res_13, mask=mask)
    tl.store(res_ptr + base_res + 8, res_20, mask=mask)
    tl.store(res_ptr + base_res + 9, res_21, mask=mask)
    tl.store(res_ptr + base_res + 10, res_22, mask=mask)
    tl.store(res_ptr + base_res + 11, res_23, mask=mask)
    tl.store(res_ptr + base_res + 12, res_30, mask=mask)
    tl.store(res_ptr + base_res + 13, res_31, mask=mask)
    tl.store(res_ptr + base_res + 14, res_32, mask=mask)
    tl.store(res_ptr + base_res + 15, res_33, mask=mask)


def mhc_bwd(
    out: torch.Tensor,
    dout: torch.Tensor,
    cg_iters: int = None,
) -> torch.Tensor:
    """Compute Sinkhorn backward using implicit CG differentiation.

    Args:
        out: Sinkhorn output R, shape (seqlen, n_stream, n_stream), float32.
        dout: Upstream gradient dR, same shape as out, float32.
        cg_iters: Number of CG iterations. Defaults to 2 * n_stream.

    Returns:
        Gradient w.r.t. pre-Sinkhorn input, same shape as out.
    """
    assert out.shape == dout.shape, "out and dout must have same shape"
    assert out.ndim == 3, "Expected 3D tensors (seqlen, n_stream, n_stream)"
    assert out.shape[1] == out.shape[2], "n_stream dimensions must match"

    seqlen, n_stream, _ = out.shape
    if cg_iters is None:
        cg_iters = 2 * n_stream

    # Ensure contiguous and float32
    out = out.contiguous().float()
    dout = dout.contiguous().float()

    # Allocate output
    res = torch.empty_like(out)

    # For n_stream=4, use optimized kernel
    if n_stream == 4:
        BLOCK_S = 64
        grid = (triton.cdiv(seqlen, BLOCK_S),)
        _mhc_bwd_kernel_n4[grid](
            out,
            dout,
            res,
            seqlen,
            cg_iters,
            BLOCK_S=BLOCK_S,
        )
    else:
        res = mhc_bwd_ref(out, dout, cg_iters=cg_iters)

    return res


def mhc_bwd_ref(
    out: torch.Tensor,
    dout: torch.Tensor,
    cg_iters: int = None,
) -> torch.Tensor:
    """PyTorch reference implementation of Sinkhorn backward via implicit CG.

    Args:
        out: Sinkhorn output R, shape (seqlen, n_stream, n_stream), float32.
        dout: Upstream gradient dR, same shape as out, float32.
        cg_iters: Number of CG iterations. Defaults to 2 * n_stream.

    Returns:
        Gradient w.r.t. pre-Sinkhorn input, same shape as out.
    """
    seqlen, n_stream, _ = out.shape
    if cg_iters is None:
        cg_iters = 2 * n_stream

    R = out.float()
    dR = dout.float()

    # RdR = R * dR
    RdR = R * dR

    # b1 = sum(RdR, dim=-1), b2 = sum(RdR, dim=-2)
    b1 = RdR.sum(dim=-1)  # (seqlen, n_stream)
    b2 = RdR.sum(dim=-2)  # (seqlen, n_stream)

    # Initialize CG
    x1 = torch.zeros_like(b1)
    x2 = torch.zeros_like(b2)

    def matvec(r, x1_in, x2_in):
        # y1[i] = sum_j(R[i,j] * x2[j]) + x1[i]
        y1 = (r * x2_in.unsqueeze(-2)).sum(dim=-1) + x1_in
        # y2[j] = sum_i(R[i,j] * x1[i]) + x2[j]
        y2 = (r * x1_in.unsqueeze(-1)).sum(dim=-2) + x2_in
        return y1, y2

    # r = b - A*x (with x=0, r = b)
    r1, r2 = b1.clone(), b2.clone()
    p1, p2 = r1.clone(), r2.clone()
    r_normsq = (r1 * r1 + r2 * r2).sum(dim=-1)  # (seqlen,)

    for _ in range(cg_iters):
        # Ap = A * p
        Ap1, Ap2 = matvec(R, p1, p2)

        # pAp = dot(p, Ap)
        pAp = (p1 * Ap1 + p2 * Ap2).sum(dim=-1)  # (seqlen,)

        # alpha = r_normsq / (pAp + eps)
        alpha = r_normsq / (pAp + EPS)
        alpha = alpha.unsqueeze(-1)  # (seqlen, 1)

        # x = x + alpha * p
        x1 = x1 + alpha * p1
        x2 = x2 + alpha * p2

        # r = r - alpha * Ap
        r1 = r1 - alpha * Ap1
        r2 = r2 - alpha * Ap2

        # r_new_normsq = dot(r, r)
        r_new_normsq = (r1 * r1 + r2 * r2).sum(dim=-1)

        # beta = r_new_normsq / (r_normsq + eps)
        beta = r_new_normsq / (r_normsq + EPS)
        beta = beta.unsqueeze(-1)

        # p = r + beta * p
        p1 = r1 + beta * p1
        p2 = r2 + beta * p2

        r_normsq = r_new_normsq

    # res = (dR - x1 - x2) * R
    res = (dR - x1.unsqueeze(-1) - x2.unsqueeze(-2)) * R
    return res


def sinkhorn_forward(
    M: torch.Tensor, iters: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn normalization forward pass.

    Args:
        M: Input logits, shape (..., n, n).
        iters: Number of Sinkhorn iterations.

    Returns:
        (R, P) where P = exp(M) and R is the doubly-stochastic matrix.
    """
    P = torch.exp(M)
    R = P.clone()
    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)
    return R, P
