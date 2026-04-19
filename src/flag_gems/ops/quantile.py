import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

from .topk import _get_finfo_val, argsort

logger = logging.getLogger(__name__)

INTERPOLATION_METHOD = ["linear", "lower", "higher", "nearest", "midpoint"]
MAX_BITONIC_M = 1024


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


@libentry()
@triton.jit
def quantile_bitonic_kernel(
    inp,
    q,
    out,
    N,
    M,
    Q,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    interpolation: tl.constexpr,
):
    pid = tle.program_id(0)
    ctype = inp.dtype.element_ty

    cols = tl.arange(0, BLOCK_M)
    mask_M = cols < M
    row_ptr = inp + pid * M
    mask_val = _get_finfo_val(ctype, return_max=True)
    vals = tl.load(row_ptr + cols, mask=mask_M, other=mask_val)
    vals = tl.where(vals.dtype.is_fp64(), vals, vals.to(tl.float32))
    ids = tl.arange(0, BLOCK_M)
    sorted_vals, _ = argsort(vals, ids, 0, descending=False)

    offsets_Q = tl.arange(0, BLOCK_Q)
    mask_Q = offsets_Q < Q
    q_vals = tl.load(q + offsets_Q, mask=mask_Q, other=0.0).to(tl.float32)
    q_scaled = q_vals * (M - 1)
    q_lower = tl.floor(q_scaled).to(tl.int32)
    q_upper = tl.ceil(q_scaled).to(tl.int32)

    idx = tl.arange(0, BLOCK_M)[:, None]
    mask_lower = idx == q_lower[None, :]
    mask_upper = idx == q_upper[None, :]
    mask_lower_f = mask_lower.to(tl.float32)
    mask_upper_f = mask_upper.to(tl.float32)
    lower_vals = tl.sum(sorted_vals[:, None] * mask_lower_f, axis=0)
    upper_vals = tl.sum(sorted_vals[:, None] * mask_upper_f, axis=0)

    if interpolation == "linear":
        q_frac = q_scaled - q_lower
        out_vals = lower_vals + (upper_vals - lower_vals) * q_frac
    elif interpolation == "lower":
        out_vals = lower_vals
    elif interpolation == "higher":
        out_vals = upper_vals
    elif interpolation == "nearest":
        q_round = tl_extra_shim.rint(q_scaled).to(tl.int32)
        out_vals = tl.where(q_round == q_upper, upper_vals, lower_vals)
    elif interpolation == "midpoint":
        out_vals = (lower_vals + upper_vals) * 0.5

    out_ptr = out + pid * Q + offsets_Q
    tl.store(out_ptr, out_vals.to(ctype), mask=mask_Q)


def quantile(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    logger.debug("GEMS QUANTILE DIM")
    assert torch.is_floating_point(inp)
    assert dim is None or isinstance(dim, int)
    assert isinstance(q, (float, torch.Tensor))
    assert interpolation in INTERPOLATION_METHOD

    # Handle dim
    if dim is None:
        inp = inp.ravel()
        dim = 0
    if dim < 0:
        dim = dim + inp.ndim

    # Handle q
    q_all_ones = False
    q_all_zeros = False
    if isinstance(q, float):
        q_all_ones = q == 1.0
        q_all_zeros = q == 0.0
        q = torch.tensor(q, device=inp.device, dtype=inp.dtype)
        Q = 1
    else:
        q = q.to(device=inp.device, dtype=inp.dtype)
        Q = 1 if q.numel() == 1 else len(q)

    assert torch.all(q >= 0.0) and torch.all(q <= 1.0)

    # Fast path: q == 0.0 -> min, q == 1.0 -> max (no sort needed)
    if q_all_ones or q_all_zeros:
        reduce_fn = torch.amax if q_all_ones else torch.amin
        if out is not None and Q == 1:
            reduce_fn(inp, dim=dim, keepdim=keepdim, out=out)
            return out
        output = reduce_fn(inp, dim=dim, keepdim=keepdim)
        if Q > 1:
            output = output.unsqueeze(0).expand(Q, *output.shape)
        if out is not None:
            out.copy_(output)
            return out
        return output

    # handle input tensor
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    M = inp.size(-1)
    N = inp.numel() // M

    output = torch.empty(inp.shape[:-1] + (Q,), dtype=inp.dtype, device=inp.device)
    if M <= MAX_BITONIC_M:
        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_Q = triton.next_power_of_2(min(Q, 16))
        grid = (N,)
        with torch_device_fn.device(inp.device):
            quantile_bitonic_kernel[grid](
                inp,
                q,
                output,
                N,
                M,
                Q,
                BLOCK_Q=BLOCK_Q,
                BLOCK_M=BLOCK_M,
                interpolation=interpolation,
            )
    else:
        sorted_vals, _ = inp.sort(dim=-1)
        grid = lambda meta: (
            triton.cdiv(Q, meta["BLOCK_Q"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        with torch_device_fn.device(inp.device):
            quantile_kernel[grid](
                sorted_vals, q, output, N, M, Q, interpolation=interpolation
            )

    if Q == 1:
        output = output.squeeze(-1)
    else:
        output = output.movedim(-1, 0)
    if keepdim:
        output = output.unsqueeze(dim + (1 if Q != 1 else 0))

    if out is not None:
        out.copy_(output)
    return output
