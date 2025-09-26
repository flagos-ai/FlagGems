import logging
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .mm import mm

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


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
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N: tl.constexpr,
    M: tl.constexpr,
    stride_an: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    buffer_size_limit: tl.constexpr,  # NOTE: `constexpr` so it can be used as a shape value.
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

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


def mv(inp, vec):
    logger.debug("GEMS MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    # TODO: fix autotune config has no item
    if M == 5333 and N == 497:
        return mv_cluster(inp, vec)

    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        if M == 1:
            mv_kernel[grid](
                inp,
                vec,
                out,
                N,
                M,
                inp.stride(0),
                inp.stride(1),
                vec.stride(0),
                out.stride(0),
                buffer_size_limit=256,
            )
        else:
            os.environ["XMLIR_MATMUL_FAST_MODE"] = "1"
            vec = vec[:, None]
            out = mm(inp, vec)
            out = out.squeeze()
            del os.environ["XMLIR_MATMUL_FAST_MODE"]
    return out


def mv_cluster(inp, vec):
    logger.debug("GEMS MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            buffer_size_limit=256,
        )
    return out
