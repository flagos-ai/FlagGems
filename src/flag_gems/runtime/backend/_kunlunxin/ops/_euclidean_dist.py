import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def _euclidean_dist_kernel_fast(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    M,
    D,
    stride_x1,
    stride_x2,
    stride_out,
    CHUNK: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fast path: D == BLOCK_D (D is a power of two) and N % BM == 0.

    No d-mask (D == BLOCK_D) and no per-row n_ok mask (N % BM == 0), so the
    backend emits unmasked vector loads. Only the [CHUNK]-shaped m_mask guards
    the M tail (M % CHUNK may be non-zero).
    """
    pid_c = tle.program_id(0)
    pid_r = tle.program_id(1)
    d = tl.arange(0, BLOCK_D)
    m = pid_c * CHUNK + tl.arange(0, CHUNK)
    m_mask = m < M
    x2_vals = tl.load(
        x2_ptr + m[:, None] * stride_x2 + d[None, :],
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    for b in tl.static_range(BM):
        n = pid_r * BM + b
        x1_vals = tl.load(x1_ptr + n * stride_x1 + d).to(tl.float32)
        diff = x1_vals[None, :] - x2_vals
        dist = tl.sqrt(tl.sum(diff * diff, axis=1))
        tl.store(out_ptr + n * stride_out + m, dist, mask=m_mask)


@libentry()
@triton.jit
def _euclidean_dist_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    M,
    D,
    stride_x1,
    stride_x2,
    stride_out,
    CHUNK: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """General path (any D / N): full d-mask and row mask."""
    pid_c = tle.program_id(0)
    pid_r = tle.program_id(1)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    m = pid_c * CHUNK + tl.arange(0, CHUNK)
    m_mask = m < M
    x2_vals = tl.load(
        x2_ptr + m[:, None] * stride_x2 + d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    for b in tl.static_range(BM):
        n = pid_r * BM + b
        n_ok = n < N
        x1_vals = tl.load(
            x1_ptr + n * stride_x1 + d,
            mask=n_ok & d_mask,
            other=0.0,
        ).to(tl.float32)
        diff = x1_vals[None, :] - x2_vals
        dist = tl.sqrt(tl.sum(diff * diff, axis=1))
        tl.store(
            out_ptr + n * stride_out + m,
            dist,
            mask=m_mask & n_ok,
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

    BM = 8
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    max_chunk = max(1, 16384 // max(BLOCK_D, 1))
    if D >= 256 or D == 64:
        CHUNK = 16
    else:
        CHUNK = 64
    CHUNK = min(CHUNK, max_chunk)

    use_fast = (D == BLOCK_D) and (N % BM == 0)
    kernel = _euclidean_dist_kernel_fast if use_fast else _euclidean_dist_kernel

    with torch_device_fn.device(x1.device):
        grid = (triton.cdiv(M, CHUNK), triton.cdiv(N, BM))
        kernel[grid](
            x1,
            x2,
            output,
            N,
            M,
            D,
            x1.stride(0),
            x2.stride(0),
            output.stride(0),
            CHUNK=CHUNK,
            BM=BM,
            BLOCK_D=BLOCK_D,
        )

    return output
