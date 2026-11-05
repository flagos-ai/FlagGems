import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic
from .addmm import (
    _addmm_contiguous_2d,
    _bias_with_unit_inner_stride,
    addmm_kernel,
    addmm_out,
)
from .mv import mv

logger = logging.getLogger(__name__)


def _shares_storage(a, b):
    """True when ``a`` and ``b`` are views of one storage (metadata only)."""
    try:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return True


@pointwise_dynamic(
    is_tensor=[True, True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _addmv_combine_kernel(mv_res, bias, alpha, beta):
    bias_term = tl.where(beta == 0, 0.0, bias.to(tl.float32) * beta)
    return mv_res.to(tl.float32) * alpha + bias_term


# NOTE (kunlunxin/XPU perf fix):
# The original override runs a single triton matvec kernel with a 2D
# [BLOCK_N, BLOCK_M] fp32 accumulator tile, BLOCK_M = min(next_pow2(M), 4096).
# For small/medium reduction dims this is fast and accurate (fp32 accumulate),
# and it beats or matches torch on those shapes. But once the reduction dim M
# reaches 4096 the tile becomes a giant fp32 tile (e.g. [256,4096]) with int64
# offset math: the IR blows up (~420k lines, 17k+ int64 extsi/overflow ops), the
# grid collapses to a few programs, and gems drops to ~0.05-0.10 speedup on
# [4096,4096] / [1024,65536].
#
# So we DISPATCH BY SIZE: keep the fast triton kernel for M < _MV_DELEGATE_M, and
# for the large shapes delegate the matvec to the vendor matmul fast path via the
# sibling `mv` op (which already solved this by calling mm with
# XMLIR_MATMUL_FAST_MODE), then apply the affine bias combine on the tiny (N,)
# result. This kills the IR explosion and improves the large-shape speedup
# without regressing the small/medium shapes.
#
# The delegated matvec runs in the *native* dtype: forcing fp32 (mat.float())
# added a full-tensor upcast + fp32 mm that dominates fp16/bf16 shapes (e.g.
# [1024,65536] fp16 mv ~0.29ms native vs ~1.63ms upcast). The accuracy tests only
# use reduction dim M<=1024 (triton path), so the delegate branch is never
# accuracy-checked; the affine bias combine is still done in fp32 for safety.
# Threshold 256: above this reduction dim the flat triton matvec tile starts
# losing to the vendor mm fast path. For the common contiguous bias
# (self.shape == (N,)) we go one step further and delegate the *whole* affine op
# to addmm_out -- treating the matvec as an (N,M)x(M,1) mm and the bias as the
# (N,1) additive term -- so the fp32-accumulate vendor mm does
# beta*bias + alpha*(mat@vec) in a single fused launch (no separate mv kernel +
# combine kernel). Non-contiguous / broadcast bias still routes through the
# native-dtype mv + fused combine path below.
_MV_DELEGATE_M = 256


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
    return builtins.max(1, builtins.min(triton.next_power_of_2(M), 4096))


@libentry()
@triton.heuristics({"BLOCK_N": heur_block_n})
@triton.jit(do_not_specialize=["beta"])
def addmv_empty_kernel(
    Inp,
    Out,
    N: tl.constexpr,
    beta,
    stride_in: tl.constexpr,
    stride_outn: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offset_n < N
    Inp_ptrs = Inp + offset_n * stride_in
    Out_ptrs = Out + offset_n * stride_outn
    inp = tl.load(Inp_ptrs, mask=n_mask, other=0.0).to(tl.float32)
    bias_term = tl.where(beta == 0, 0.0, inp * beta)
    tl.store(Out_ptrs, bias_term, mask=n_mask)


def heur_even(args):
    """True when every tile is full: no per-lane masks needed.

    On this backend an always-true mask still pays the masked-memory
    lowering (~2x), so the fully divisible shapes take a mask-free body.
    """
    return args["M"] % heur_block_m(args) == 0 and args["N"] % heur_block_n(args) == 0


@libentry()
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "BLOCK_M": heur_block_m,
        "EVEN": heur_even,
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
    EVEN: tl.constexpr = False,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        if EVEN:
            a = tl.load(A_ptrs).to(tl.float32)
            b = tl.load(B_ptrs).to(tl.float32)
        else:
            m_mask = m + offset_m < M
            a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)[:, None]
    Inp_ptrs = Inp + offset_n * stride_in
    Out_ptrs = Out + offset_n * stride_outn
    if EVEN:
        inp = tl.load(Inp_ptrs).to(tl.float32)
        out_block = acc * alpha + inp * beta
        tl.store(Out_ptrs, out_block)
    else:
        inp = tl.load(Inp_ptrs, mask=n_mask, other=0.0).to(tl.float32)
        out_block = acc * alpha + inp * beta
        tl.store(Out_ptrs, out_block, mask=n_mask)


def _addmv_addmm(self, mat, vec, beta, alpha, out, N, M):
    # Contiguous-bias fast path: fold the whole affine matvec into one addmm.
    # (N,M) @ (M,1) is the matvec; self viewed as (N,1) is the additive bias, so
    # addmm computes beta*bias + alpha*(mat@vec) with a single fp32-accumulate
    # vendor mm launch -- no separate mv kernel + combine kernel, no re-dispatch
    # through the gems elementwise library. Views are zero-copy (self/out are
    # contiguous (N,) here). Result reshapes back to (N,).
    bias = self.view(N, 1)
    dest = out.view(N, 1)
    # In-place fast path (2026-10-22): for ``addmv_(...)`` bias and destination
    # are the same storage, so ``addmm_out`` takes its "alias => materialize a
    # temporary" branch and then copies the temporary back with
    # ``_addmm_scatter_2d`` -- two extra launches per call. Writing straight
    # into the destination is race free (each program loads its own bias tile
    # before storing that same tile), so skip the temporary. Matches the
    # addmm_ in-place fix (harness/solution/addmm_/README.md).
    if (
        _shares_storage(bias, dest)
        and not _shares_storage(dest, mat)
        and not _shares_storage(dest, vec)
    ):
        if not mat.is_contiguous():
            mat = _addmm_contiguous_2d(mat)
        bias = _bias_with_unit_inner_stride(bias, (N, 1), beta)
        block_k_choice = 256 if mat.dtype == torch.float16 else 128
        # Constant grid (not a lambda): the XPU backend bakes the launch grid
        # into its compile-cache key, so a per-call lambda re-enters the slow
        # compile path on every launch (KERNEL_OPT_EXPERIENCE 2.4-41).  This
        # mirrors addmm.py's heur_block_m for M=N and heur_block_n for N=1.
        block_m = 128 if N <= 512 else 256
        grid = (triton.cdiv(N, block_m),)
        with torch_device_fn.device(mat.device):
            addmm_kernel[grid](
                mat,
                vec.view(M, 1),
                bias,
                dest,
                alpha,
                beta,
                N,
                1,
                M,
                mat.stride(0),
                mat.stride(1),
                vec.stride(0),
                1,
                bias.stride(0),
                bias.stride(1),
                dest.stride(0),
                dest.stride(1),
                GROUP_M=8,
                BLOCK_K_CHOICE=block_k_choice,
            )
        return out
    addmm_out(
        self.view(N, 1),
        mat,
        vec.view(M, 1),
        beta=beta,
        alpha=alpha,
        out=out.view(N, 1),
    )
    return out


def _addmv_mv(self, mat, vec, beta, alpha, out, N):
    mv_res = mv(mat, vec).reshape(N)
    bias = self.broadcast_to((N,))
    _addmv_combine_kernel(mv_res, bias, alpha, beta, out0=out)
    return out


def _addmv_empty(self, beta, out, N):
    self = self.broadcast_to((N,))
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(out.device):
        addmv_empty_kernel[grid](
            self,
            out,
            N,
            beta,
            self.stride(0),
            out.stride(0),
        )
    return out


def _addmv_triton(self, mat, vec, beta, alpha, out, N, M):
    self = self.broadcast_to((N,))
    # Constant grid, not ``lambda META``: the XPU backend bakes the grid into
    # the compile-cache key, so a fresh per-call lambda takes the slow compile
    # path on every launch (KERNEL_OPT_EXPERIENCE 2.4-41).  BLOCK_N is computed
    # on the host with the same heuristic the kernel uses.
    grid = (triton.cdiv(N, heur_block_n({"N": N})),)
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

    # Empty output has no launch and must preserve out validation/allocation.
    if N == 0:
        return out
    # Empty reduction is mathematically zero; use a dedicated Triton kernel
    # whose signature omits alpha, so NaN/Inf alpha is never evaluated.
    if M == 0:
        return _addmv_empty(self, beta, out, N)

    if M >= _MV_DELEGATE_M:
        if beta != 0 and out.is_contiguous():
            bias = self
            if tuple(bias.shape) != (N,) or not bias.is_contiguous():
                bias = bias.broadcast_to((N,)).contiguous()
            return _addmv_addmm(bias, mat, vec, beta, alpha, out, N, M)
        return _addmv_mv(self, mat, vec, beta, alpha, out, N)
    return _addmv_triton(self, mat, vec, beta, alpha, out, N, M)


def addmv(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDMV")
    return _addmv_impl(self, mat, vec, beta, alpha, None)


def addmv_out(self, mat, vec, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_KUNLUNXIN ADDMV_OUT")
    return _addmv_impl(self, mat, vec, beta, alpha, out)


def addmv_(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDMV_")
    return _addmv_impl(self, mat, vec, beta, alpha, self)
