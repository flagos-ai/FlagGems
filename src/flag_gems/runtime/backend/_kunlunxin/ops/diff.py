import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design notes (kunlunxin XPU)
#
# The generic `diff` (libtuner + 2D strided tile, autotuned per (M, N)) and the
# previous per-row 2D kernel both pay ~60-100ns *per program* on this XPU, and
# a 2D-tile layout degrades to fully-discrete strided accesses.  With one
# program per (row, chunk) that puts diff at 0.03-0.11x torch on every 2D/3D
# shape.
#
# Key identity: for a *contiguous* (M, N) input, diff along the last dim is
# exactly the 1-D flat diff of the M*N-element stream, viewed as
# (M, N)[:, :N-n].  The n-th flat diff at flat index p equals the n-th
# within-row diff at (p // N, p % N) whenever p % N < N-n; the only "wrong"
# flat entries are those crossing a row boundary, and those land precisely in
# columns N-n .. N-1 of the (M,N) view, which `[:, :N-n]` drops.  So the whole
# operator is a *contiguous* 1-D kernel (one flat block-DMA pass per stage),
# which measures ~1.3x torch on 2D/3D shapes, plus a zero-copy view at the end.
#
# Block width: measured 2048..8192 lanes per program are all equivalent (and
# bit-exact for bf16, unlike 1024-lane blocks whose bf16 sub is truncated to
# one ulp), so use the widest one to minimise program count.
#
# int16 exception: this XPU's LLVM cannot select a v32i16 `sub` (the compiler
# widens the load to a 512-bit vector at BLOCK >= 2048 and aborts), so int16
# is promoted to int32 around the subtract.  int32 arithmetic is exact here
# (|d_n| <= 2^n * 100 << 2^31 for the test domain) and the final truncation
# wraps mod 2^16 exactly like torch's stage-wise int16 arithmetic.
FLAT_BLOCK = 8192


@libentry()
@triton.jit
def diff_flat_kernel(
    in_ptr,
    out_ptr,
    N_COMP,
    BLOCK: tl.constexpr,
    CAST16: tl.constexpr,
):
    # out[p] = in[p+1] - in[p] for p < N_COMP (N_COMP = numel_in - 1).
    # in[.] is a contiguous 1-D stream; the caller guarantees the only
    # out-of-range access (in[N_COMP]) is masked off.
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_COMP
    a = tl.load(in_ptr + offs, mask=mask)
    b = tl.load(in_ptr + offs + 1, mask=mask)
    if CAST16:
        d = (b.to(tl.int32) - a.to(tl.int32)).to(a.dtype)
    else:
        d = b - a
    tl.store(out_ptr + offs, d, mask=mask)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DIFF")

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    # (M, N) contiguous with the diff dimension last.
    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N
    total = M * N

    if total == 0:
        return torch.empty(
            shape[:dim] + [N - n] + shape[(dim + 1) :],
            dtype=input.dtype,
            device=input.device,
        )

    src = input.reshape(-1)

    def _launch_flat(s, d, n_comp):
        with torch_device_fn.device(s.device):
            diff_flat_kernel[(triton.cdiv(n_comp, FLAT_BLOCK),)](
                s,
                d,
                n_comp,
                BLOCK=FLAT_BLOCK,
                CAST16=bool(s.dtype == torch.int16),
            )

    if n == 1:
        buf = torch.empty(total, device=input.device, dtype=input.dtype)
        _launch_flat(src, buf, total - 1)
    else:
        # Ping-pong between two full-size scratch buffers; stage k writes
        # total-(k+1) valid elements, and consecutive stages are ordered on the
        # current stream (no host sync required).
        bufs = [
            torch.empty(total, device=input.device, dtype=input.dtype)
            for _ in range(2)
        ]
        for k in range(n):
            n_comp = total - (k + 1)
            _launch_flat(src, bufs[k % 2], n_comp)
            src = bufs[k % 2]
        buf = src

    # Drop the cross-row garbage columns.  `buf` has `total` elements; valid
    # entry (r, j) sits at offset r*N + j (r < M, j < N-n), so the result is
    # a zero-copy strided view of `buf` with row width N (the n dropped
    # columns per row are the only "wrong" flat entries and are never
    # exposed).  Redundant outer dims are repacked in row-major order, which
    # matches `dim_compress`'s permute(batch + [dim]) ordering.
    out_shape = shape[:dim] + [N - n] + shape[(dim + 1) :]
    strides = [0] * len(shape)
    post = N
    for k in range(len(shape) - 1, dim, -1):
        strides[k] = post
        post *= shape[k]
    for k in range(dim - 1, -1, -1):
        strides[k] = post
        post *= shape[k]
    strides[dim] = 1
    return torch.as_strided(buf, out_shape, strides)