# Kunlunxin (XPU) override of _euclidean_dist.
#
# _euclidean_dist(x1, x2) computes pairwise Euclidean distances:
#   out[i, j] = ||x1[i] - x2[j]||_2 ,  x1:(N,D)  x2:(M,D)  out:(N,M)
#
# The generic KernelGen kernel (src/flag_gems/ops/_euclidean_dist.py) launches
# grid=(N, M): ONE program per output element, each re-loading the full D-length
# rows of BOTH x1 and x2, doing a D-reduction and storing a single scalar. On XPU
# that is launch-bound (N*M tiny programs) + O(N*M*D) redundant x1 reloads ->
# [128,256]x[128,256] (=32768 programs) gems speedup ~0.02, [64,128] ~0.075
# (harness/perf_ir_3/ir-euclidean_dist-dev6.log).
#
# Fix: keep the 1D-reduction structure (XPU handles 1D tiles well; a 2D
# [BLOCK_M,BLOCK_D] tile + axis reduction hits `out of resource: uni_sram`, and a
# gems-op composition -- matmul + norms + elementwise -- chains ~10 kernel launches
# at ~0.15ms each and can wedge the device via async double-buffering), but:
#   1) load the x1 row ONCE per program and reuse it across a CHUNK of x2 rows
#      (kills the redundant x1 reloads), and
#   2) have each program own a CHUNK of output columns so the launch count drops
#      from N*M to N*cdiv(M,CHUNK).
# CHUNK is picked to keep the grid around a few hundred programs (enough XPU
# parallelism without over-serializing each program).
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
    BLOCK_D: tl.constexpr,
):
    pid_n = tle.program_id(0)
    pid_mc = tle.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D
    # Load the x1 row once; reuse across all CHUNK output columns.
    x1_vals = tl.load(
        x1_ptr + pid_n * stride_x1 + d_offsets, mask=d_mask, other=0.0
    ).to(tl.float32)

    for i in range(CHUNK):
        m = pid_mc * CHUNK + i
        m_ok = m < M
        x2_vals = tl.load(
            x2_ptr + m * stride_x2 + d_offsets,
            mask=d_mask & m_ok,
            other=0.0,
        ).to(tl.float32)
        diff = x1_vals - x2_vals
        dist = tl.sqrt(tl.sum(diff * diff))
        if m_ok:
            tl.store(out_ptr + pid_n * stride_out + m, dist)


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

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    # Target ~512 programs total: split M into cdiv(512, N) column blocks.
    n_col_blocks = max(1, triton.cdiv(512, N))
    CHUNK = triton.cdiv(M, n_col_blocks)

    with torch_device_fn.device(x1.device):
        grid = (N, triton.cdiv(M, CHUNK))
        _euclidean_dist_kernel[grid](
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
            BLOCK_D=BLOCK_D,
        )

    return output
