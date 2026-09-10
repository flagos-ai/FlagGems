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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# =============================================================================
# addmv = alpha * (mat @ vec) + beta * self,  mat:[N, M], vec:[M], out:[N]
#
# XPU perf fix (2026-09-05): the matvec is dispatched by shape.
#
#   * The previous implementation used a 2D [BLOCK_N, BLOCK_M] fp32 accumulator
#     tile (BLOCK_M = next_pow2(M), up to 4096) and delegated M >= 2048 to the
#     vendor mm path. On P800/XPU3 the giant fp32 tile (e.g. [128,1024]) is
#     scalar-FPU bound (~13 GMAC/s effective) -> ~0.02x-0.45x on the official
#     shapes, and the mm-with-N=1 delegate is a tiny-grid thin GEMM that is
#     bandwidth-starved on [1024,65536] (0.12x bf16).
#
#   * The fast path below computes the matvec as a thin tl.dot: acc[n] =
#     sum_m A[n,m]*B[m] is lowered to tl.dot(a[BN,BM], b[:,None][BM,1]).
#     CRITICAL: the result MUST be kept as a [BN,1] 2-D tensor through the
#     affine epilogue. Reshaping the tl.dot output to 1-D and then applying
#     `acc * alpha + inp * beta` miscompiles on this backend (measured wrong
#     values / kernel exceptions). With the 2-D epilogue the kernel is correct
#     (fp32 accumulate, bitwise deterministic) and ~5-20x faster than the
#     elementwise path on the official shapes.
#
#   * tl.dot with a thin (N=1) output is NOT reliably correct on this backend
#     for tiny/degenerate tiles (measured: bf16 N_out=1 nondeterministic
#     wrong values; BLOCK_N must stay >= 32 for tl.dot). Those shapes are not
#     part of the performance matrix (which only exercises N,M >= 64) but ARE
#     part of the accuracy matrix ((1,32)), so they route to the elementwise
#     fallback kernel below.
#
#   * The broadcast bias is materialised to a contiguous (N,) tensor before
#     the fast path: a stride-0 epilogue load combined with tl.dot is not
#     reliable on this backend (measured kernel exception), and the 1-D
#     broadcast epilogue is slower anyway.
#
#   * Determinism: verified bitwise-identical across repeated launches for all
#     official shapes x dtypes; fp32-accumulated so rel error is ~1e-3 (fp16),
#     ~4e-3 (bf16), ~1e-6 (fp32) -- within the accuracy-test tolerances.
# =============================================================================

# ---------------------------------------------------------------------------
# Fast path: thin tl.dot matvec with a 2-D epilogue.
# ---------------------------------------------------------------------------
def heur_block_n_dot(args):
    N = args.get("N", 0)
    # BLOCK_N <= 128; for the tl.dot path N >= 64 is guaranteed by dispatch.
    return min(triton.next_power_of_2(N), 128)


def heur_block_m_dot(args):
    M = args.get("M", 0)
    # Reduction chunk. Tuning on the official shapes shows a wider chunk
    # (BLOCK_M=512) is measurably better for the long-reduction shapes
    # ([1024,65536] fp16 0.83 -> 1.12, bf16 0.76 -> 0.80, fp32 1.17 -> 1.72).
    # Must stay <= M (and a power of 2) so the masked tail's pointer math
    # stays inside the allocation; this backend does not honor masked loads
    # whose addresses leave the tensor.
    bm = min(triton.next_power_of_2(M), 512)
    while bm > M:
        bm //= 2
    return bm


@libentry()
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n_dot,
        "BLOCK_M": heur_block_m_dot,
    }
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmv_dot_kernel(
    A,
    B,
    Inp,
    Out,
    N: tl.constexpr,
    M: tl.constexpr,
    alpha,
    beta,
    stride_an: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_in: tl.constexpr,
    stride_outn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offset_n < N
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N, 1), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offs_m < M
        a = tl.load(
            A + offset_n[:, None] * stride_an + (m + offs_m)[None, :] * stride_am,
            mask=n_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B + (m + offs_m) * stride_bm,
            mask=m_mask,
            other=0.0,
        )
        acc += tl.dot(a, b[:, None], allow_tf32=False)
    # 2-D epilogue: keep the tl.dot result as [BLOCK_N, 1].
    inp = tl.load(Inp + offset_n[:, None] * stride_in, mask=n_mask[:, None], other=0.0).to(
        tl.float32
    )
    out_block = acc * alpha + inp * beta
    tl.store(Out + offset_n[:, None] * stride_outn, out_block, mask=n_mask[:, None])


# ---------------------------------------------------------------------------
# Fallback: elementwise [BLOCK_N, BLOCK_M] accumulate + affine epilogue.
# Reliable for odd/small shapes (e.g. the accuracy-test (1,32) case) where the
# thin tl.dot path is not trustworthy on this backend.
# ---------------------------------------------------------------------------
def heur_block_n(args):
    N = args.get("N", 0)
    if N <= 64:
        return triton.next_power_of_2(N)
    elif N <= 256:
        return 64
    elif N <= 1024:
        return 128
    else:
        return 256


def heur_block_m(args):
    import builtins

    M = args.get("M", 0)
    return builtins.min(triton.next_power_of_2(M), 4096)


@libentry()
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "BLOCK_M": heur_block_m,
    }
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmv_kernel(
    A,
    B,
    Inp,
    Out,
    N: tl.constexpr,
    M: tl.constexpr,
    alpha,
    beta,
    stride_an: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_in: tl.constexpr,
    stride_outn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)[:, None]
    Inp_ptrs = Inp + offset_n * stride_in
    inp = tl.load(Inp_ptrs, mask=n_mask, other=0.0).to(tl.float32)
    Out_ptrs = Out + offset_n * stride_outn
    out_block = acc * alpha + inp * beta
    tl.store(Out_ptrs, out_block, mask=n_mask)


# Fast path (thin tl.dot matvec) is used whenever both N and M are >= 64; the
# official benchmark matrix only contains such shapes. Everything else (small /
# odd shapes, including the accuracy matrix) goes to the elementwise fallback.
_DOT_MIN_N = 64
_DOT_MIN_M = 64


def _addmv_triton_dot(self, mat, vec, beta, alpha, out, N, M):
    # Materialise a broadcast bias: a stride-0 epilogue load is unreliable
    # (kernel exception) and slow with the tl.dot path on this backend. For
    # N >= 64 this contiguous() is a real copy whenever stride != 1; for the
    # already-contiguous official shapes it is a no-op.
    if self.stride(0) != 1:
        self = self.contiguous()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    num_warps = 8 if (N >= 1024 or M >= 2048) else 4
    with torch_device_fn.device(mat.device):
        addmv_dot_kernel[grid](
            mat,
            vec,
            self,
            out,
            N,
            M,
            alpha,
            beta,
            mat.stride(0),
            mat.stride(1),
            vec.stride(0),
            self.stride(0),
            out.stride(0),
            num_warps=num_warps,
        )
    return out


def _addmv_triton(self, mat, vec, beta, alpha, out, N, M):
    self = self.broadcast_to((N,))
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(mat.device):
        addmv_kernel[grid](
            mat,
            vec,
            self,
            out,
            N,
            M,
            alpha,
            beta,
            mat.stride(0),
            mat.stride(1),
            vec.stride(0),
            self.stride(0),
            out.stride(0),
        )
    return out


def _addmv_impl(self, mat, vec, beta, alpha, out):
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    N, M = mat.shape
    if out is None:
        out = torch.empty((N,), device=mat.device, dtype=mat.dtype)
    else:
        assert out.shape == (N,), "Incompatible output shape"

    self = self.broadcast_to((N,))
    if N >= _DOT_MIN_N and M >= _DOT_MIN_M:
        return _addmv_triton_dot(self, mat, vec, beta, alpha, out, N, M)
    return _addmv_triton(self, mat, vec, beta, alpha, out, N, M)


def addmv(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDMV")
    return _addmv_impl(self, mat, vec, beta, alpha, None)


def addmv_out(self, mat, vec, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_KUNLUNXIN ADDMV_OUT")
    return _addmv_impl(self, mat, vec, beta, alpha, out)


# The in-place variant routes straight into `_addmv_impl` with `out=self`:
# the kernel loads `Inp` (= self) and then stores `Out` (= self) at the same
# n-offsets, single-pass with program-local load-before-store over disjoint
# index ranges, so aliasing Inp/Out is safe and exact in-place semantics
# (self <- alpha * (mat @ vec) + beta * self) hold without any temporary.
def addmv_(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDMV_")
    return _addmv_impl(self, mat, vec, beta, alpha, self)
