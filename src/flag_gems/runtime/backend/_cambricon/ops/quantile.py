import logging

import torch
import triton
import triton.language as tl
import triton.language.core as core
from torch import Tensor

try:
    # TODO: Triton 2.1 does not implement _log2.
    # Remove the try-catch block once all vendors upgrade to a newer version of Triton.
    from triton.language.standard import _log2, zeros_like
except ImportError:
    pass
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

from ..utils import MAX_GRID_SIZE_X
from .topk import _get_finfo_val

logger = logging.getLogger(__name__)
logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

INTERPOLATION_METHOD = ["linear", "lower", "higher", "nearest", "midpoint"]
MAX_BITONIC_M = 1024

"""
Note(Zhengzekang):
Refer from triton2.2 official `sort` implementation:
https://github.com/triton-lang/triton/blob/release/2.2.x/python/triton/language/standard.py#L392-L404
Just add indices to sort with values.
"""


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    # tl.device_print("shape is: ", shape)
    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # actual compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


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
    grid_0 = tl.num_programs(0)
    ctype = inp.dtype.element_ty

    while pid < N:
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
        pid += grid_0


def quantile(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    logger.debug("GEMS_CAMBRICON QUANTILE DIM")
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
        grid = min(N, MAX_GRID_SIZE_X // 4)
        with torch_device_fn.device(inp.device):
            quantile_bitonic_kernel[(grid,)](
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
