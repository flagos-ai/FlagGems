import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic
from .addmm import addmm_out
from .mv import mv

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _addmv_combine_kernel(mv_res, bias, alpha, beta):
    return mv_res.to(tl.float32) * alpha + bias.to(tl.float32) * beta


#
#
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


def _addmv_addmm(self, mat, vec, beta, alpha, out, N, M):
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
    bias = torch.zeros_like(mv_res) if beta == 0 else self.broadcast_to((N,))
    _addmv_combine_kernel(mv_res, bias, alpha, beta, out0=out)
    return out


def _addmv_triton(self, mat, vec, beta, alpha, out, N, M):
    if beta == 0:
        self = torch.zeros_like(self)
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
        out = torch.empty(N, device=mat.device, dtype=mat.dtype)
    else:
        assert out.shape == (N,), "Incompatible output shape"

    if M == 0:
        if beta == 0:
            out.zero_()
        else:
            out.copy_(self.broadcast_to((N,)).mul(beta))
        return out

    if M >= _MV_DELEGATE_M:
        if (
            beta != 0
            and tuple(self.shape) == (N,)
            and self.is_contiguous()
            and out.is_contiguous()
        ):
            return _addmv_addmm(self, mat, vec, beta, alpha, out, N, M)
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
