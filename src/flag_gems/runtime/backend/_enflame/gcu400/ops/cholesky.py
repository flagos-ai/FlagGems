# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# Capture the native ``aten::cholesky`` kernel at import time, before FlagGems
# registers its own implementation on the device dispatch key. Calling the
# captured handle later bypasses the dispatcher entirely, so the fallback for
# matrix orders beyond the Triton kernel's supported range never re-enters
# FlagGems -- a plain ``torch.cholesky`` call here would recurse into this very
# function. Guarded for older PyTorch versions without ``torch.library.get_kernel``.
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")

if _CAN_CAPTURE_NATIVE_KERNEL:
    _NATIVE_CHOLESKY = torch.library.get_kernel(
        torch.ops.aten.cholesky.default, runtime.device.dispatch_key
    )
    _DEVICE_KEYSET = torch._C.DispatchKeySet(
        getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
    )


@libentry()
@triton.jit
def cholesky_kernel(
    A_ptr,
    Out_ptr,
    Info_ptr,
    N,
    s0,
    s1,
    s2,
    o0,
    o1,
    o2,
    UPPER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NSTEPS: tl.constexpr,
    IS64: tl.constexpr,
):
    """Blocked Cholesky decomposition kernel.

    One program per matrix: the whole ``BLOCK_N x BLOCK_N`` tile stays
    register-resident and is factored with an unblocked right-looking
    Cholesky using rank-1 updates.  ``Info_ptr`` receives the 1-based index
    of the first non-positive pivot (0 on success), matching the
    ``aten::linalg_cholesky_ex`` convention.
    """
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK_N)
    ri = r[:, None]
    cj = r[None, :]
    valid = (ri < N) & (cj < N)

    # --- load one whole matrix, then symmetrize ----------------------------
    x = tl.load(A_ptr + pid * s0 + ri * s1 + cj * s2, mask=valid, other=0.0)
    if IS64:
        W = x
    else:
        W = x.to(tl.float32)

    # symmetry: tri + tri^T would double the diagonal
    tri = tl.where(ri >= cj, W, 0.0)
    W = tri + tl.trans(tri) - tl.where(ri == cj, tri, 0.0)
    W = tl.where(valid, W, 0.0)

    # --- unblocked right-looking Cholesky (rank-1 updates) ------------------
    # Invariant: entering step j the tile holds
    #     W = A - sum_{k<j} L[:, k] L[:, k]^T
    # on the trailing (i>j, c>j) sub-block.  The trip count is the exact matrix
    # order, not the padded tile width: rows/columns >= N are zero and the store
    # is masked to the valid region, so the padded steps would be dead work on
    # the serial critical path.
    #
    # The loop is deliberately rolled (runtime ``range``) instead of
    # ``tl.static_range``: unrolling every step duplicates the whole tile worth
    # of temporaries, which pushes BLOCK_N=64 past the GCU DSM/local limit and
    # spills to local memory (measured 1.29ms -> 0.50ms when rolled at N=64,
    # 0.35ms -> 0.22ms at N=32).
    info = 0
    for j in range(0, NSTEPS):
        col = tl.sum(tl.where(cj == j, W, 0.0), axis=1)
        pivot = tl.sum(tl.where(r == j, col, 0.0))
        # A non-positive or non-finite pivot means the leading minor of order
        # j+1 is not positive definite.  Record the first such j and keep the
        # loop going: the trip count is a static range, and the trailing values
        # are discarded by the caller once info != 0.
        bad = (pivot <= 0.0) or (pivot != pivot)
        if bad and info == 0:
            info = j + 1
        scale = tl.sqrt(pivot)
        inv = 1.0 / scale
        col = tl.where(r > j, col * inv, 0.0)
        col = tl.where(r == j, scale, col)
        # column j now holds L[:, j] exactly (zeros above the diagonal)
        W = tl.where(cj == j, col[:, None], W)
        upd = col[:, None] * col[None, :]
        W = tl.where((ri > j) & (cj > j), W - upd, W)

    tl.store(
        Out_ptr + pid * o0 + ri * o1 + cj * o2,
        W.to(Out_ptr.dtype.element_ty),
        mask=valid,
    )
    tl.store(Info_ptr + pid, info)


_MAX_N = 32

# Cache of the one-element info buffer used by the single-matrix hot path.
# Keyed by device; the vector is fully overwritten by every launch, so reuse
# cannot leak state between calls.
_INFO_CACHE = {}


def cholesky(A, upper=False):
    logger.debug("GEMS_ENFLAME CHOLESKY")
    if A.dim() < 2:
        raise RuntimeError(
            "cholesky: The input tensor A must have at least 2 dimensions."
        )
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), "cholesky only supports float32 and float64"
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "cholesky: A must be batches of square matrices, but they are "
            "%d by %d matrices" % (A.shape[-2], A.shape[-1])
        )

    n = int(A.shape[-1])

    # A zero-sized matrix is vacuously factorised; ATen returns a fresh empty
    # output rather than aliasing the input.
    if n == 0 or A.numel() == 0:
        return torch.empty_like(A)

    if A.dtype == torch.float64:
        # The GCU compiler cannot lower float64 arithmetic in this kernel
        # (``PassManager execution failed`` even for a 2x2 tile), while the
        # native LAPACK-backed ``aten::cholesky`` does support float64 on GCU.
        # Hand float64 over to it instead of failing to compile.
        if _CAN_CAPTURE_NATIVE_KERNEL:
            return _NATIVE_CHOLESKY.call_boxed(_DEVICE_KEYSET, A, bool(upper))
        raise NotImplementedError("cholesky float64 requires the native aten kernel")

    if n > _MAX_N:
        # The blocked kernel keeps the whole matrix live for the entire serial
        # factorisation, which is only viable for small orders. Beyond that,
        # delegate to the native LAPACK-backed ``aten::cholesky`` captured at
        # import time; raising here would break callers (e.g. the benchmark)
        # since a registered device kernel has no automatic ATen fallback.
        # The cutoff is 32 because the Triton path is slower than that native
        # path as soon as the tile stops fitting the register/DSM budget:
        # measured 0.59x at N=64 versus 0.92-1.04x at N<=32.
        if _CAN_CAPTURE_NATIVE_KERNEL:
            return _NATIVE_CHOLESKY.call_boxed(_DEVICE_KEYSET, A, bool(upper))
        raise NotImplementedError("cholesky kernel supports N <= 32")
    num = int(A.numel()) // (n * n)

    # A contiguous copy is required: reshaping a transposed input would copy
    # silently and lose writes, and the kernel indexes with explicit strides.
    a2 = A.reshape(num, n, n).contiguous()
    out = torch.empty_like(a2)
    # The single-matrix case is the hot path and always needs a one-element info
    # buffer, so keep it cached per device instead of paying an allocator round
    # trip on every call (measured ~15us of the wrapper budget).
    info = None
    if num == 1:
        info = _INFO_CACHE.get(A.device)
        if info is None:
            info = torch.empty(1, dtype=torch.int32, device=A.device)
            _INFO_CACHE[A.device] = info
    if info is None:
        info = torch.empty(num, dtype=torch.int32, device=A.device)

    block_n = triton.next_power_of_2(n)
    # A single warp is fastest on GCU400: every extra warp replicates the
    # BLOCK_N x BLOCK_N tile footprint in DSM, so num_warps=2 regressed N=32
    # from 0.24ms to 0.30ms and N=64 from 0.50ms to 0.65ms, while num_warps=4
    # at BLOCK_N=64 overflows the 896KB DSM budget outright.
    num_warps = 1
    cholesky_kernel[(num,)](
        a2,
        out,
        info,
        n,
        a2.stride(0),
        a2.stride(1),
        a2.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        UPPER=bool(upper),
        BLOCK_N=block_n,
        NSTEPS=n,
        IS64=bool(A.dtype == torch.float64),
        num_warps=num_warps,
    )

    # The info read-back is a fixed host-side synchronisation cost on every call,
    # so keep it on the cheapest path: the previous `torch.any(info != 0).item()`
    # paid 41us extra (two reduction kernels plus a sync) and raised the GCU400
    # wrapper latency at N=2/8/32 from 34.6/52.3/142.4us to 54.2/71.5/161.7us.
    # Reading a single element for num==1 needs one DtoH transfer and no
    # reduction kernel; for batches read `max` first as a fast check and only
    # locate the first failing index on the rare error path.
    if num == 1:
        first = int(info[0].item())
    else:
        first = int(info.max().item())
        if first != 0:
            first = int(info[int(torch.nonzero(info.reshape(-1))[0])].item())
    if first != 0:
        raise torch.linalg.LinAlgError(
            "cholesky: The factorization could not be completed because the "
            "input is not positive-definite (the leading minor of order %d is "
            "not positive-definite)." % first
        )

    if upper:
        # ATen returns the upper factor as a transposed view of the freshly
        # allocated buffer; mirror that instead of forcing a contiguous copy.
        out = out.mT.conj()

    return out.reshape(A.shape)


__all__ = ["cholesky"]
