import logging

import torch

from flag_gems.ops.scatter import scatter_

logger = logging.getLogger(__name__)


def _normalize_reduce(reduce: str) -> str:
    if reduce is None:
        raise ValueError("reduce must be specified for scatter_reduce")
    reduce = reduce.lower()
    if reduce in ("sum", "add"):
        return "add"
    if reduce in ("prod", "product", "mul", "multiply"):
        return "multiply"
    if reduce in ("amax", "max"):
        return "amax"
    if reduce in ("amin", "min"):
        return "amin"
    if reduce == "mean":
        return "mean"
    raise ValueError(f"Unsupported reduce type: {reduce}")


def _scatter_count(
    template: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    count_dtype = torch.float32
    counts = torch.zeros_like(template, dtype=count_dtype)
    ones = torch.ones_like(index, dtype=count_dtype, device=index.device)
    scatter_(counts, dim, index, ones, reduce="add")
    return counts


def _scatter_reduce_impl(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    include_self: bool,
) -> torch.Tensor:
    dim = dim % inp.dim()
    reduce = _normalize_reduce(reduce)
    src_strided = src.as_strided(index.shape, src.stride())

    work_dtype = inp.dtype
    cast_back = False
    if inp.dtype == torch.bfloat16:
        work_dtype = torch.float32
        cast_back = True

    if work_dtype != inp.dtype:
        inp_work = inp.to(work_dtype)
        src_work = src_strided.to(work_dtype)
    else:
        inp_work = inp
        src_work = src_strided

    out = inp_work.clone()

    if reduce == "mean":
        counts = _scatter_count(inp_work, dim, index)
        if include_self:
            sum_out = out
            count_out = torch.ones_like(out, dtype=counts.dtype)
        else:
            hit = counts > 0
            sum_out = torch.where(hit, torch.zeros_like(out), out)
            count_out = torch.where(
                hit,
                torch.zeros_like(out, dtype=counts.dtype),
                torch.ones_like(out, dtype=counts.dtype),
            )

        scatter_(sum_out, dim, index, src_work, reduce="add")
        ones = torch.ones_like(index, dtype=counts.dtype, device=index.device)
        scatter_(count_out, dim, index, ones, reduce="add")
        out = sum_out / count_out.to(sum_out.dtype)
    else:
        if not include_self:
            counts = _scatter_count(inp_work, dim, index)
            hit = counts > 0
            if reduce == "add":
                out = torch.where(hit, torch.zeros_like(out), out)
            elif reduce == "multiply":
                out = torch.where(hit, torch.ones_like(out), out)
            elif reduce == "amax":
                fill_val = torch.finfo(out.dtype).min
                out = torch.where(hit, torch.full_like(out, fill_val), out)
            elif reduce == "amin":
                fill_val = torch.finfo(out.dtype).max
                out = torch.where(hit, torch.full_like(out, fill_val), out)

        scatter_(out, dim, index, src_work, reduce=reduce)

    if cast_back:
        out = out.to(inp.dtype)
    return out


def scatter_reduce(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
):
    logger.debug("GEMS SCATTER_REDUCE")
    return _scatter_reduce_impl(inp, dim, index, src, reduce, include_self)


def scatter_reduce_(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
):
    logger.debug("GEMS SCATTER_REDUCE_")
    out = _scatter_reduce_impl(inp, dim, index, src, reduce, include_self)
    inp.copy_(out)
    return inp


def scatter_reduce_out(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
    out: torch.Tensor,
):
    logger.debug("GEMS SCATTER_REDUCE_OUT")
    result = _scatter_reduce_impl(inp, dim, index, src, reduce, include_self)
    out.copy_(result)
    return out
