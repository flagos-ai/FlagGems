import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

INTERPOLATION_METHOD = ["linear", "lower", "higher", "nearest", "midpoint"]


def heur_block_q(args):
    return triton.next_power_of_2(min(triton.cdiv(args["Q"], 8), 16))


def heur_block_n(args):
    if args["N"] >= 65536:
        return triton.next_power_of_2(triton.cdiv(args["N"], 512))
    elif args["N"] >= 4096:
        return triton.next_power_of_2(triton.cdiv(args["N"], 128))
    elif args["N"] >= 64:
        return 32
    elif args["N"] >= 32:
        return 4
    else:
        return 1


@libentry()
@triton.heuristics(values={"BLOCK_Q": heur_block_q, "BLOCK_N": heur_block_n})
@triton.jit
def quantile_kernel(
    inp,
    q,
    out,
    N,
    M,
    Q,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    interpolation: tl.constexpr,
):
    pid_Q = tle.program_id(0)
    pid_N = tle.program_id(1)
    ctype = inp.dtype.element_ty

    offsets_Q = pid_Q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_Q = offsets_Q < Q
    q_ptrs = q + offsets_Q

    offsets_N = pid_N * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_N = offsets_N < N

    out_ptrs = out + offsets_N[:, None] * Q + offsets_Q[None, :]
    mask_out = mask_N[:, None] & mask_Q[None, :]

    q_block = tl.load(q_ptrs, mask_Q, 0.0).to(ctype) * (M - 1)
    q_lower = tl.floor(q_block).to(tl.int32)
    q_upper = tl.ceil(q_block).to(tl.int32)

    inp_lower = tl.load(
        inp + offsets_N[:, None] * M + q_lower[None, :], mask_N[:, None], 0.0
    )
    inp_upper = tl.load(
        inp + offsets_N[:, None] * M + q_upper[None, :], mask_N[:, None], 0.0
    )

    if interpolation == "linear":
        q_frac = q_block - q_lower
        tl.store(out_ptrs, inp_lower + (inp_upper - inp_lower) * q_frac, mask_out)

    elif interpolation == "lower":
        tl.store(out_ptrs, inp_lower, mask_out)

    elif interpolation == "higher":
        tl.store(out_ptrs, inp_upper, mask_out)

    elif interpolation == "nearest":
        q_round = tl_extra_shim.rint(q_block)
        out_block = tl.where(q_round == q_upper, inp_upper, inp_lower)
        tl.store(out_ptrs, out_block, mask_out)

    elif interpolation == "midpoint":
        tl.store(out_ptrs, (inp_lower + inp_upper) / 2, mask_out)


def quantile(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    logger.debug("GEMS QUANTILE DIM")
    assert torch.is_floating_point(inp)
    assert dim is None or isinstance(dim, int)
    assert isinstance(q, (float, torch.Tensor))
    assert interpolation in INTERPOLATION_METHOD

    M = inp.numel()
    if isinstance(q, float):
        q = torch.tensor(q, device=inp.device)
        Q = 1
    else:
        Q = 1 if q.numel() == 1 else len(q)

    assert M > 0
    assert Q > 0
    assert torch.all(q >= 0.0) and torch.all(q <= 1.0)

    if dim is None:
        inp = inp.ravel()
        dim = 0

    shape = list(inp.shape)

    dim %= inp.ndim
    inp = dim_compress(inp, dim)
    M = shape[dim]
    N = inp.numel() // M

    inp, _ = inp.sort()  # Sort the input with torch.sort()
    output = torch.empty(inp.shape[:-1] + (Q,), dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(Q, meta["BLOCK_Q"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(inp.device):
        quantile_kernel[grid](inp, q, output, N, M, Q, interpolation=interpolation)

    output = output.permute(
        (-1,) + tuple(range(0, inp.ndim - 1))
    )  # Same as torch.quantile()
    if keepdim:
        output = output.unsqueeze(dim + 1)
    if Q == 1:
        output = output.squeeze(0)

    if out is not None:
        out.copy_(output)
    return output
