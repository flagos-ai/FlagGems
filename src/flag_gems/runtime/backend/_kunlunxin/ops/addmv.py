import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def heur_block_n(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))


def heur_block_m(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["M"]), 4096)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("mv"),
#     key=["M", "N"],
# )
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
    N,
    M,
    alpha,
    beta,
    stride_an,
    stride_am,
    stride_bm,
    stride_in,
    stride_outn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tle.program_id(0)
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


def addmv(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMV")
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    N, M = mat.shape
    out = torch.empty((N,), device=mat.device, dtype=mat.dtype)
    self = self.broadcast_to(out.shape)
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


def addmv_out(self, mat, vec, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS ADDMV OUT")
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    N, M = mat.shape
    if out is None:
        out = torch.empty((N,), device=mat.device, dtype=mat.dtype)
    else:
        assert out.shape == (N,), "Incompatible output shape"

    self = self.broadcast_to(out.shape)
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
