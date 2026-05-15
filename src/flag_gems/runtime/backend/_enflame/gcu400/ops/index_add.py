import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def index_add_row_kernel(
    src,
    index,
    out,
    alpha,
    M,
    N_src,
    N_out,
    ALPHA_ONE: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("src,")
            code.writeline("out,")
            code.writeline("N,")
            code.writeline("inp_numel,")
            code.writeline("inp_stride_dim,")
            code.writeline("inp_shape_dim,")
            code.writeline("src_shape_dim,")
            code.writeline("delta,")
            code.writeline("alpha,")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"src_shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for src")

            code.writeline("BLOCK_SIZE: tl.constexpr,")

        code.writeline("):")

            out_off = out_base + idx
            tl.atomic_add(out + out_off, val, mask=col_mask, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def index_add_flat_kernel(
    src,
    index,
    out,
    alpha,
    N_total,
    N_src,
    N_out,
    ALPHA_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N_total

    row = offset // N_src
    col = offset % N_src

    idx = tl.load(index + col, mask=mask, other=0).to(tl.int32)
    src_val = tl.load(src + offset, mask=mask, other=0.0)

    if ALPHA_ONE:
        val = src_val
    else:
        val = src_val * alpha

    out_off = row * N_out + idx
    tl.atomic_add(out + out_off, val, mask=mask, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def index_add_non_inner_kernel(
    src,
    index,
    out,
    alpha,
    M,
    N_src,
    N_out,
    K,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_work = M * N_src

    with code.indent():
        code.writeline("src_strides = list(src.stride())")
        code.writeline("src_shapes = list(src.shape)")

        # kernel launch
        code.writeline("BLOCK_SIZE = 256")  # BLOCK_SIZE setting
        code.writeline("if N // BLOCK_SIZE > 65535:")
        with code.indent():
            code.writeline("BLOCK_SIZE = 16384")
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline(
                "index, src, out, N, inp_numel, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, alpha, "
            )
            if rank > 0:
                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE")
        code.writeline(")")
        code.writeline("return out")

        for k_off in tl.range(0, K, BLOCK_K):
            k = k_off + tl.arange(0, BLOCK_K)
            k_mask = k < K

            src_val = tl.load(src + src_base + k, mask=k_mask, other=0.0)
            tl.atomic_add(
                out + out_base + k, src_val * alpha, mask=k_mask, sem="relaxed"
            )


def _select_dispatch(M, N_src, N_total):
    if M >= 256 and N_src >= 64:
        BLOCK_COL = min(triton.next_power_of_2(N_src), 2048)
        num_programs = min(M, 48)
        return "row", BLOCK_COL, num_programs, 4
    else:
        if N_total <= 4096:
            bs = 128
        elif N_total <= 131072:
            bs = 256
        else:
            bs = 1024
        grid_size = min(triton.cdiv(N_total, bs), 65535)
        nw = 4
        return "flat", bs, grid_size, nw


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX ADD")
    if index.dtype == torch.int64:
        index = index.to(torch.int32)
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    out = inp.clone()

    M = math.prod(inp.shape[:dim]) if dim > 0 else 1
    N_src = src.size(dim)
    N_out = inp.size(dim)
    K = math.prod(inp.shape[dim + 1 :]) if dim < inp.ndim - 1 else 1

    if N_src == 0:
        return out

    alpha_one = alpha == 1

    if K == 1:
        N_total = M * N_src
        mode, block_param, grid_param, nw = _select_dispatch(M, N_src, N_total)

        if mode == "row":
            with torch_device_fn.device(inp.device):
                index_add_row_kernel[(grid_param,)](
                    src,
                    index,
                    out,
                    alpha,
                    M,
                    N_src,
                    N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_COL=block_param,
                    num_warps=nw,
                )
        else:
            with torch_device_fn.device(inp.device):
                index_add_flat_kernel[(grid_param,)](
                    src,
                    index,
                    out,
                    alpha,
                    N_total,
                    N_src,
                    N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_SIZE=block_param,
                    num_warps=nw,
                )
    else:
        BLOCK_K = min(triton.next_power_of_2(K), 1024)
        total_work = M * N_src
        num_programs = min(total_work, 48)

        with torch_device_fn.device(inp.device):
            index_add_non_inner_kernel[(num_programs,)](
                src,
                index,
                out,
                alpha,
                M,
                N_src,
                N_out,
                K,
                BLOCK_K=BLOCK_K,
                num_warps=4,
            )

    _index_add_func(
        out,
        index,
        src,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
        alpha,
    )
    return out

def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX ADD_")
    if index.dtype == torch.int64:
        index = index.to(torch.int32)
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    if not inp.is_contiguous():
        result = index_add(inp, dim, index, src, alpha)
        inp.copy_(result)
        return inp

    src = src.contiguous()

    M = math.prod(inp.shape[:dim]) if dim > 0 else 1
    N_src = src.size(dim)
    N_out = inp.size(dim)
    K = math.prod(inp.shape[dim + 1 :]) if dim < inp.ndim - 1 else 1

    if N_src == 0:
        return inp

    alpha_one = alpha == 1

    if K == 1:
        N_total = M * N_src
        mode, block_param, grid_param, nw = _select_dispatch(M, N_src, N_total)

        if mode == "row":
            with torch_device_fn.device(inp.device):
                index_add_row_kernel[(grid_param,)](
                    src,
                    index,
                    inp,
                    alpha,
                    M,
                    N_src,
                    N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_COL=block_param,
                    num_warps=nw,
                )
        else:
            with torch_device_fn.device(inp.device):
                index_add_flat_kernel[(grid_param,)](
                    src,
                    index,
                    inp,
                    alpha,
                    N_total,
                    N_src,
                    N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_SIZE=block_param,
                    num_warps=nw,
                )
    else:
        BLOCK_K = min(triton.next_power_of_2(K), 1024)
        total_work = M * N_src
        num_programs = min(total_work, 48)

        with torch_device_fn.device(inp.device):
            index_add_non_inner_kernel[(num_programs,)](
                src,
                index,
                inp,
                alpha,
                M,
                N_src,
                N_out,
                K,
                BLOCK_K=BLOCK_K,
                num_warps=4,
            )

    _index_add_func(
        inp,
        index,
        src,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
        alpha,
    )
    return inp
