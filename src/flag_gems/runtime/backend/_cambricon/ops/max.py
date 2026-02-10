import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.limits import get_dtype_min

from ..utils import TOTAL_CORE_NUM, cfggen_reduce_op, prune_reduce_config

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def max_kernel_float_once(
    inp,
    out,
    M: tl.constexpr,
):
    offset = tl.arange(0, M)
    inp_val = tl.load(inp + offset)
    max_val = tl.max(inp_val, 0)
    tl.store(out, max_val)


@libentry()
@libtuner(
    configs=cfggen_reduce_op(),
    key=["M"],
    strategy=["log"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    }
)
@triton.jit
def max_kernel_float(
    inp, out, M, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    res = -float("inf")

    if ONE_TILE_PER_CTA:
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=-float("inf"))
        (res,) = tl.max(inp_val, 0, return_indices=True)
        tl.atomic_max(out, res)
    else:
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        _tmp = tl.full([BLOCK_SIZE], value=-float("inf"), dtype=inp.dtype.element_ty)
        for off in range(block_start, M, step):
            offset = off + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=-float("inf"))
            _tmp = tl.where((inp_val > _tmp), inp_val, _tmp)
        (res,) = tl.max(_tmp, 0, return_indices=True)
        tl.atomic_max(out, res)


@libentry()
@libtuner(
    configs=cfggen_reduce_op(),
    key=["M"],
    strategy=["log"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    }
)
@triton.jit
def max_kernel_int(
    inp, out, M, FILL_VALUE, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    res = FILL_VALUE
    if ONE_TILE_PER_CTA:
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=FILL_VALUE)
        res = tl.max(inp_val)
    else:
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        block_start = pid * BLOCK_SIZE
        _tmp = tl.full([BLOCK_SIZE], value=-(2**63), dtype=tl.int64)
        for off in range(block_start, M, step):
            offset = off + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=FILL_VALUE)
            _tmp = tl.where((inp_val > _tmp), inp_val, _tmp)
        res = tl.max(_tmp)
    tl.atomic_max(out, res)


@libentry()
@libtuner(
    configs=cfggen_reduce_op(),
    key=["M"],
    strategy=["log"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    }
)
@triton.jit
def max_kernel_int64_1(
    inp, mid, M, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    FILL_VALUE = -(2**63)
    res = FILL_VALUE
    if ONE_TILE_PER_CTA:
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=FILL_VALUE)
        res = tl.max(inp_val)
    else:
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        block_start = block_start.to(tl.int64)
        _tmp = tl.full([BLOCK_SIZE], value=FILL_VALUE, dtype=tl.int64)
        for off in range(block_start, M, step):
            offset = off + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=FILL_VALUE)
            _tmp = tl.where((inp_val > _tmp), inp_val, _tmp)
        res = tl.max(_tmp)
    tl.store(mid + pid, res)


@libentry()
@triton.jit
def max_kernel_int64_2(mid, out, BLOCK_NUM: tl.constexpr):
    offset = tl.arange(0, BLOCK_NUM)
    mid_val = tl.load(mid + offset)
    out_val = tl.max(mid_val)
    tl.store(out, out_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("max"),
    key=[
        "M",
        "N",
    ],
    strategy=["log", "log"],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UPCAST: tl.constexpr = False,
):
    # set offset
    if UPCAST:
        pid_m = tl.program_id(0).to(tl.int64)
        pid_k = tl.program_id(1).to(tl.int64)
    else:
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    result_value = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    min_value = get_dtype_min(inp.type.element_ty)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # set mask
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


def max(inp):
    logger.debug("GEMS_CAMBRICON MAX")
    inp = inp.contiguous()
    M = inp.numel()
    dtype = inp.dtype
    device = inp.device
    mid_size = TOTAL_CORE_NUM
    grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_SIZE"]), mid_size),)

    with torch_device_fn.device(inp.device):
        if torch.is_floating_point(inp):
            if M <= 65536:
                out = torch.empty([], dtype=dtype, device=device)
                max_kernel_float_once[(1, 1, 1)](inp, out, M)
            else:
                out = torch.full([], float("-inf"), dtype=torch.float32, device=device)
                max_kernel_float[grid](inp, out, M)
        elif dtype == torch.int64:
            mid = torch.empty([mid_size], dtype=dtype, device=device)
            out = torch.empty([], dtype=dtype, device=device)
            # Because atomic op don't support i64, use two kernels.
            max_kernel_int64_1[(mid_size, 1, 1)](inp, mid, M, enable_soft_i64=True)
            max_kernel_int64_2[(1, 1, 1)](
                mid, out, BLOCK_NUM=mid_size, enable_soft_i64=True
            )
        else:
            fill_value = torch.iinfo(dtype).min
            out = torch.full([], -(2**31), dtype=torch.int32, device=device)
            max_kernel_int[grid](inp, out, M, fill_value)
    return out.to(dtype)


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_CAMBRICON MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)
    UPCAST = inp.shape[0] * inp.stride(0) >= 1 << 31

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(inp.device):
        max_kernel[grid](inp, out_value, out_index, M, N, K, UPCAST=UPCAST)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out
