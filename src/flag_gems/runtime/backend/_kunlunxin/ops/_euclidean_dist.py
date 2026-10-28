import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _euclidean_dist_mm_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    M,
    D,
    stride_x1,
    stride_x2,
    stride_out,
    BN: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """dist = sqrt(relu(||x1||^2 + ||x2||^2 - 2 x1.x2^T)), the same GEMM identity
    torch._euclidean_dist uses. The x1.x2^T term goes through tl.dot with
    input_precision=ieee (full fp32 -- tf32 is ~2 off and fails the 1.3e-6 rtol
    gate); the squared norms are reduced alongside in the same K loop.

    NOTE: the final combine is written step-by-step (acc, then += n1, then += n2)
    on purpose -- the fused `n1[:,None] + n2[None,:] - 2*dot` expression
    miscompiles on this backend (two broadcasts of reduction vectors in one
    expression -> wrong result; verified), whereas the stepwise form is exact.
    """
    pid_n = tle.program_id(0)
    pid_m = tle.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = pid_m * BM + tl.arange(0, BM)
    nmask = rn < N
    mmask = rm < M
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    n1 = tl.zeros((BN,), dtype=tl.float32)
    n2 = tl.zeros((BM,), dtype=tl.float32)
    for k in range(0, D, BK):
        kk = k + tl.arange(0, BK)
        kmask = kk < D
        a = tl.load(
            x1_ptr + rn[:, None] * stride_x1 + kk[None, :],
            mask=nmask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            x2_ptr + rm[:, None] * stride_x2 + kk[None, :],
            mask=mmask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
        n1 += tl.sum(a * a, axis=1)
        n2 += tl.sum(b * b, axis=1)
    dist2 = -2.0 * acc
    dist2 = dist2 + n1[:, None]
    dist2 = dist2 + n2[None, :]
    dist = tl.sqrt(tl.where(dist2 > 0.0, dist2, 0.0))
    tl.store(
        out_ptr + rn[:, None] * stride_out + rm[None, :],
        dist,
        mask=nmask[:, None] & mmask[None, :],
    )


def _euclidean_dist(x1, x2):
    logger.debug("GEMS_KUNLUNXIN _EUCLIDEAN_DIST")

    assert x1.ndim == 2, "x1 must be a 2D tensor"
    assert x2.ndim == 2, "x2 must be a 2D tensor"
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same number of columns"

    N, D = x1.shape
    M = x2.shape[0]

    x1 = x1.contiguous()
    x2 = x2.contiguous()
    output = torch.empty((N, M), dtype=x1.dtype, device=x1.device)

    if N == 0 or M == 0:
        return output

    # Small output tiles spread the [N,M] grid across many programs/cores (the
    # pair counts here are tiny and otherwise launch-bound); BK >= 16 so tl.dot
    # is legal even for tiny D, with kk < D masking the pad lanes to zero.
    BN = 32
    BM = 16
    BK = max(16, min(triton.next_power_of_2(D), 256))

    with torch_device_fn.device(x1.device):
        grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
        _euclidean_dist_mm_kernel[grid](
            x1,
            x2,
            output,
            N,
            M,
            D,
            x1.stride(0),
            x2.stride(0),
            output.stride(0),
            BN=BN,
            BM=BM,
            BK=BK,
        )

    return output
