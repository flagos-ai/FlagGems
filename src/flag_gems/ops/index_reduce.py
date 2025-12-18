import importlib
import logging
import os
from typing import Any, Callable, Mapping, Optional, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("import torch")
    code.newline()
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")

    code.newline()
    code.newline()
    return code


def generate_index_reduce_kernel(
    rank: int,
    kernel_name: str,
    reduce_op: str,
    dtype: torch.dtype,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("out,")
        code.writeline("index,")
        code.writeline("src,")
        code.writeline("dim: tl.constexpr,")
        code.writeline("inp_stride_dim,")
        code.writeline("inp_shape_dim,")
        code.writeline("src_shape_dim,")
        code.writeline("delta,")
        code.writeline("N,")
        code.writeline("inp_numel,")
        code.writeline("include_self: tl.constexpr,")

        if reduce_op == "mean":
            code.writeline("count_tensor,")

        for i in range(rank):
            code.writeline(f"src_stride_{i},")
            code.writeline(f"src_shape_{i},")

        code.writeline("BLOCK_SIZE: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(axis=0)")
        code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        code.writeline("mask = offsets < N")

        for i in range(rank - 1, -1, -1):
            code.writeline(f"src_offset{i} = offsets % src_shape_{i}")
            code.writeline(f"offsets = offsets // src_shape_{i}")
        code.newline()
        comp = [f"src_offset{i} * src_stride_{i}" for i in range(rank)]
        code.writeline(f"src_offset = {' + '.join(comp)}")

        code.writeline("pre_cal = (inp_stride_dim * src_shape_dim)")

        code.writeline("pre_idx = (src_offset // pre_cal).to(tl.int64)")
        code.writeline(
            "dim_idx = (src_offset % pre_cal // inp_stride_dim).to(tl.int64)"
        )
        code.writeline(
            "src_dim_idx = (tl.load(index + dim_idx, mask=mask, other=0)).to(tl.int64)"
        )
        code.writeline(
            'assert src_dim_idx >= 0 and src_dim_idx < inp_shape_dim, "0 <= index < self.size(dim)"'
        )
        code.writeline(
            "input_idx = (src_offset + (delta * pre_idx + src_dim_idx - dim_idx) * inp_stride_dim).to(tl.int64)"
        )

        code.writeline("input_mask = input_idx < inp_numel")
        code.writeline("valid_mask = mask & input_mask")

        other_val = "0.0"
        code.writeline(
            "add_on = tl.load(src + src_offset, mask=valid_mask, other={})".format(
                other_val
            )
        )

        if reduce_op == "prod":
            # if atomic op is not support in triton, we implement it by atomic_CAS
            code.writeline("while tl.sum(valid_mask) > 0:")
            with code.indent():
                code.writeline(
                    "old_val = tl.load(out + input_idx, mask=valid_mask, other=1.0)"
                )
                code.writeline(
                    "new_val = tl.where(valid_mask, old_val * add_on, old_val)"
                )
                code.writeline(
                    "prev_val = tl.where(valid_mask, tl.atomic_cas(out + input_idx, old_val, new_val), old_val)"
                )
                code.writeline("success = (prev_val == old_val) | (~valid_mask)")
                code.writeline("valid_mask = valid_mask & (~success)")
        elif reduce_op == "amax":
            if dtype == torch.float32:
                code.writeline(
                    "tl.atomic_max(out + input_idx, add_on, mask=valid_mask, sem='relaxed')"
                )
            else:  # triton not supporting torch.fp16 in atomic_max and atomic_min now, so implement by CAS is needed
                code.writeline("while tl.sum(valid_mask) > 0:")
                with code.indent():
                    code.writeline(
                        "old_val = tl.load(out + input_idx, mask=valid_mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where(valid_mask, tl.maximum(old_val, add_on), old_val)"
                    )
                    code.writeline(
                        "prev_val = tl.where(valid_mask, tl.atomic_cas(out + input_idx, old_val, new_val), old_val)"
                    )
                    code.writeline("success = (prev_val == old_val) | (~valid_mask)")
                    code.writeline("valid_mask = valid_mask & (~success)")
        elif reduce_op == "amin":
            if dtype == torch.float32:
                code.writeline(
                    "tl.atomic_min(out + input_idx, add_on, mask=valid_mask, sem='relaxed')"
                )
            else:
                code.writeline("while tl.sum(valid_mask) > 0:")
                with code.indent():
                    code.writeline(
                        "old_val = tl.load(out + input_idx, mask=valid_mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where(valid_mask, tl.minimum(old_val, add_on), old_val)"
                    )
                    code.writeline(
                        "prev_val = tl.where(valid_mask, tl.atomic_cas(out + input_idx, old_val, new_val), old_val)"
                    )
                    code.writeline("success = (prev_val == old_val) | (~valid_mask)")
                    code.writeline("valid_mask = valid_mask & (~success)")
        elif reduce_op == "mean":
            code.writeline(
                "tl.atomic_add(out + input_idx, add_on, mask=valid_mask, sem='relaxed')"
            )
            code.writeline(
                "tl.atomic_add(count_tensor + input_idx, 1.0, mask=valid_mask, sem='relaxed')"
            )

    code.newline()
    code.newline()
    return code


def generate_mean_post_process_kernel(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("@triton.jit")
    code.writeline("def _mean_post_process_kernel(")
    with code.indent():
        code.writeline("out,")
        code.writeline("count_tensor,")
        code.writeline("numel,")
        code.writeline("include_self: tl.constexpr,")
        code.writeline("BLOCK_SIZE: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        code.writeline("mask = offsets < numel")

        code.writeline("sum_val = tl.load(out + offsets, mask=mask)")
        code.writeline("count_val = tl.load(count_tensor + offsets, mask=mask)")

        code.writeline("if include_self:")
        with code.indent():
            code.writeline("adjusted_count = count_val + 1.0")
        code.writeline("else:")
        with code.indent():
            code.writeline("adjusted_count = count_val")

        code.writeline(
            "mean_val = tl.where(adjusted_count > 0, sum_val / adjusted_count, 0.0)"
        )
        code.writeline("tl.store(out + offsets, mean_val, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_destination_passing_wrapper(
    rank: int, wrapper_name: str, kernel_name: str, reduce_op: str, code: IndentedBuffer
) -> IndentedBuffer:
    parameters = [
        "out",
        "index",
        "src",
        "dim",
        "inp_stride_dim",
        "inp_shape_dim",
        "src_shape_dim",
        "delta",
        "N",
        "inp_numel",
        "include_self",
        "reduce",
    ]

    wrapper_signature = f"def {wrapper_name}({', '.join(parameters)}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("addr = out.data_ptr()")
        code.writeline("BLOCK_SIZE = 128")
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")

        if reduce_op == "mean":
            code.writeline("count_tensor = torch.zeros_like(out)")

        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            code.writeline("out,")
            code.writeline("index,")
            code.writeline("src,")
            code.writeline("dim,")
            code.writeline("inp_stride_dim,")
            code.writeline("inp_shape_dim,")
            code.writeline("src_shape_dim,")
            code.writeline("delta,")
            code.writeline("N,")
            code.writeline("inp_numel,")
            code.writeline("include_self,")

            if reduce_op == "mean":
                code.writeline("count_tensor,")

            for i in range(rank):
                code.writeline(f"src.stride({i}),")
                code.writeline(f"src.size({i}),")

            code.writeline("BLOCK_SIZE,")
        code.writeline(")")

        if reduce_op == "mean":
            code.writeline("post_grid = (triton.cdiv(inp_numel, BLOCK_SIZE),)")
            code.writeline("_mean_post_process_kernel[post_grid](")
            with code.indent():
                code.writeline("out,")
                code.writeline("count_tensor,")
                code.writeline("inp_numel,")
                code.writeline("include_self,")
                code.writeline("BLOCK_SIZE,")
            code.writeline(")")
        code.writeline(
            "assert addr == out.data_ptr(), 'The output tensor has been reallocated, which is not allowed.'"
        )
        code.writeline("return out")

    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs  [out, index, src, dim, inp_stride_dim, inp_shape_dim, src_shape_dim, delta,
    # N, inp_numel, include_self, reduce]
    shape = inputs[2].shape
    reduce_op = inputs[-1]
    rank = len(shape)
    inp_dtype = inputs[0].dtype

    code = generate_imports(code)
    code = generate_index_reduce_kernel(rank, kernel_name, reduce_op, inp_dtype, code)

    if reduce_op == "mean":
        code = generate_mean_post_process_kernel(code)

    code = generate_destination_passing_wrapper(
        rank, wrapper_name, kernel_name, reduce_op, code
    )
    return code


class IndexReduceFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self._arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_index_reduce_wrapper",
                "_index_reduce_jit_function",
                code,
            )

            file_name = f"index_reduce_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_reduce_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def _arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        reduce_op = args[-1]
        include_self = args[-2]
        dtype = args[0].dtype
        key = f"{max_rank}_{reduce_op}_{int(include_self)}_{str(dtype)}"
        return key


_index_reduce_func = IndexReduceFunction()


def index_reduce(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
    out: Optional[torch.Tensor] = None,
):
    """
    Perform an index-based reduction operation on the input tensor along a specified dimension.
    """
    logger.debug("GEMS INDEX_REDUCE")
    assert ((0 <= index) & (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"
    assert reduce in [
        "prod",
        "mean",
        "amax",
        "amin",
    ], "reduce should be one of 'prod', 'mean', 'amax', 'amin'"
    assert (
        inp.dtype == src.dtype
    ), "index_reduce_(): inp and source must have the same scalar type"

    if out is not None:
        assert out.shape == inp.shape, "out tensor must have the same shape as inp"
        assert out.device == inp.device, "out tensor must be on the same device as inp"
        assert out.dtype == inp.dtype, "out tensor must have the same dtype as inp"
        result_tensor = out
    else:
        result_tensor = torch.empty_like(inp)

    if include_self:
        result_tensor.copy_(inp)
    else:
        if reduce == "prod":
            result_tensor.fill_(1.0)
        elif reduce == "mean":
            result_tensor.fill_(0.0)
        elif reduce == "amax":
            result_tensor.fill_(float("-inf"))
        elif reduce == "amin":
            result_tensor.fill_(float("inf"))
        else:
            result_tensor.fill_(0.0)

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    _index_reduce_func(
        result_tensor,
        index,
        src,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
        include_self,
        reduce,
    )
    return result_tensor


def index_reduce_(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
):
    """
    In-place version of index_reduce.
    """
    logger.debug("GEMS INDEX_REDUCE_")
    assert ((0 <= index) & (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"
    assert reduce in [
        "prod",
        "mean",
        "amax",
        "amin",
    ], "reduce should be one of 'prod', 'mean', 'amax', 'amin'"

    if not include_self:
        if reduce == "prod":
            inp.fill_(1.0)
        elif reduce == "mean":
            inp.fill_(0.0)
        elif reduce == "amax":
            inp.fill_(float("-inf"))
        elif reduce == "amin":
            inp.fill_(float("inf"))
        else:
            inp.fill_(0.0)

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    _index_reduce_func(
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
        include_self,
        reduce,
    )
    return inp
