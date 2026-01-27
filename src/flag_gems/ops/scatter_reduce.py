"""Scatter Reduce operator implementation using Triton kernels."""
import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    code.newline()
    code.newline()
    return code


def generate_scatter_reduce_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    # Heuristic functions
    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return 4")
    code.newline()
    code.newline()

    # Decorators
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")

    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{src_stride_vars},{shape_vars}])"
    )

    # Kernel signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("src_strided,")
            code.writeline("index,")
            code.writeline("inp,")
            code.writeline("out,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            code.writeline("IS_SUM: tl.constexpr,")
            code.writeline("IS_PROD: tl.constexpr,")
            code.writeline("IS_MEAN: tl.constexpr,")
            code.writeline("IS_AMAX: tl.constexpr,")
            code.writeline("IS_AMIN: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    # Kernel body
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")

            # Calculate offsets for each dimension
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                    code.writeline(f"src_stride_{i} = src_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                code.writeline(f"src_offsets += mod * src_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            # Load values
            code.writeline(
                "cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)"
            )
            code.writeline(
                "cur_index = tl.load(index + idx_offsets, mask=mask, other=0)"
            )
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
                code.writeline("stride_dim = stride_dim.to(tl.int32)")

            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.newline()

            # Apply reduction operation
            code.writeline("if IS_SUM:")
            with code.indent():
                code.writeline(
                    "tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
                )

            code.writeline("elif IS_PROD:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_out = tl.load(out + inp_offsets, mask=mask, other=1.0)"
                    )
                    code.writeline(
                        "new_val = tl.where(stop, cur_out, cur_out * cur_src)"
                    )
                    code.writeline(
                        "cas_result = tl.atomic_cas(out + inp_offsets, cur_out, new_val, sem='relaxed')"
                    )
                    code.writeline("stop |= cur_out == cas_result")
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("elif IS_MEAN:")
            with code.indent():
                code.writeline(
                    "tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
                )

            code.writeline("elif IS_AMAX:")
            with code.indent():
                code.writeline(
                    "# Use compare-and-swap for max (fp16/bf16 not supported by atomic_max)"
                )
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_out = tl.load(out + inp_offsets, mask=mask, other=float('-inf'))"
                    )
                    code.writeline(
                        "# Convert to float32 for comparison, then back to original dtype"
                    )
                    code.writeline("cur_out_f32 = cur_out.to(tl.float32)")
                    code.writeline("cur_src_f32 = cur_src.to(tl.float32)")
                    code.writeline(
                        "new_val_f32 = tl.where(stop, cur_out_f32, tl.maximum(cur_out_f32, cur_src_f32))"
                    )
                    code.writeline("new_val = new_val_f32.to(cur_out.dtype)")
                    code.writeline(
                        "cas_result = tl.atomic_cas(out + inp_offsets, cur_out, new_val, sem='relaxed')"
                    )
                    code.writeline("# Compare in float32 for consistency")
                    code.writeline("cas_result_f32 = cas_result.to(tl.float32)")
                    code.writeline("stop |= cur_out_f32 == cas_result_f32")
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("elif IS_AMIN:")
            with code.indent():
                code.writeline(
                    "# Use compare-and-swap for min (fp16/bf16 not supported by atomic_min)"
                )
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_out = tl.load(out + inp_offsets, mask=mask, other=float('inf'))"
                    )
                    code.writeline(
                        "# Convert to float32 for comparison, then back to original dtype"
                    )
                    code.writeline("cur_out_f32 = cur_out.to(tl.float32)")
                    code.writeline("cur_src_f32 = cur_src.to(tl.float32)")
                    code.writeline(
                        "new_val_f32 = tl.where(stop, cur_out_f32, tl.minimum(cur_out_f32, cur_src_f32))"
                    )
                    code.writeline("new_val = new_val_f32.to(cur_out.dtype)")
                    code.writeline(
                        "cas_result = tl.atomic_cas(out + inp_offsets, cur_out, new_val, sem='relaxed')"
                    )
                    code.writeline("# Compare in float32 for consistency")
                    code.writeline("cas_result_f32 = cas_result.to(tl.float32)")
                    code.writeline("stop |= cur_out_f32 == cas_result_f32")
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    parameters: List[str] = []
    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")
    parameters.append("reduce: str")
    parameters.append("int32_offset: bool = True")
    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = list(inp.stride())")
        code.writeline("index_strides = index.stride()")
        code.writeline("src_strides = src_strided.stride()")
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("inp_size_dim = dim_size")
        code.writeline("stride_dim = dim_stride")

        code.writeline('IS_SUM = reduce == "sum"')
        code.writeline('IS_PROD = reduce == "prod"')
        code.writeline('IS_MEAN = reduce == "mean"')
        code.writeline('IS_AMAX = reduce == "amax"')
        code.writeline('IS_AMIN = reduce == "amin"')

        # Kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]),')
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("src_strided, index, inp, out,")
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("inp_size_dim,")
                code.writeline("stride_dim,")
                code.writeline("N,")
                code.writeline("IS_SUM,")
                code.writeline("IS_PROD,")
                code.writeline("IS_MEAN,")
                code.writeline("IS_AMAX,")
                code.writeline("IS_AMIN,")
                code.writeline("INT32_OFFSET=int32_offset,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_reduce_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterReduceFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_scatter_reduce_wrapper",
                "_scatter_reduce_jit_function",
                code,
            )

            file_name = f"scatter_reduce_rank_{key}_pid_{self.pid}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # Load module
            spec = importlib.util.spec_from_file_location(
                f"_gen_scatter_reduce_rank_{key}_pid_{self.pid}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_reduce_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_reduce_func = ScatterReduceFunction()


def scatter_reduce(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    """
    Reduces all values from src into input at the indices specified in index.

    Args:
        input: The input tensor
        dim: The dimension along which to index
        index: The indices of elements to scatter
        src: The source tensor
        reduce: The reduction operation ("sum", "prod", "mean", "amax", "amin")
        include_self: Whether to include input values in the reduction

    Returns:
        Output tensor with reduced values
    """
    logger.debug("GEMS SCATTER_REDUCE")

    # Validate inputs
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim-1}], but got {dim})"
        )

    dim = dim % input.ndim

    if index.ndim != input.ndim:
        raise RuntimeError(
            f"Index tensor must have the same number of dimensions as input tensor "
            f"({input.ndim}), but got {index.ndim}"
        )

    if src.ndim != input.ndim:
        raise RuntimeError(
            f"Source tensor must have the same number of dimensions as input tensor "
            f"({input.ndim}), but got {src.ndim}"
        )

    for d in range(input.ndim):
        if d != dim:
            if index.size(d) > input.size(d):
                raise RuntimeError(
                    f"Index tensor size at dimension {d} ({index.size(d)}) "
                    f"must not exceed input size ({input.size(d)})"
                )
            if index.size(d) != src.size(d):
                raise RuntimeError(
                    f"Index and source must have the same size at dimension {d}, "
                    f"but got {index.size(d)} and {src.size(d)}"
                )

    if reduce not in ["sum", "prod", "mean", "amax", "amin"]:
        raise ValueError(
            f"reduce argument must be one of 'sum', 'prod', 'mean', 'amax', 'amin', "
            f"but got '{reduce}'"
        )

    # Initialize output
    output = input.clone()

    # For include_self=False, we need to reset scattered positions to identity values
    # before performing the scatter operation
    if not include_self:
        # Create a mask of which positions in output will be scattered to
        # by scattering ones to those positions
        scatter_mask = torch.zeros_like(output, dtype=torch.bool)

        # Use index to mark which positions will be scattered to
        # We need to handle this carefully for each dimension
        if reduce == "sum" or reduce == "mean":
            identity_value = 0.0
        elif reduce == "prod":
            identity_value = 1.0
        elif reduce == "amax":
            identity_value = (
                float("-inf")
                if output.dtype.is_floating_point
                else torch.iinfo(output.dtype).min
            )
        elif reduce == "amin":
            identity_value = (
                float("inf")
                if output.dtype.is_floating_point
                else torch.iinfo(output.dtype).max
            )

        # Scatter identity values to positions that will be updated
        # First, mark which positions will be scattered to
        ones_like_src = torch.ones_like(src, dtype=torch.bool)
        scatter_mask = torch.scatter(scatter_mask, dim, index, ones_like_src)

        # Set those positions to identity values
        output = torch.where(
            scatter_mask, torch.full_like(output, identity_value), output
        )

    # For float16 sum/mean operations, use float32 accumulation to improve precision
    # This reduces rounding errors from atomic operations
    use_fp32_accum = output.dtype == torch.float16 and reduce in ["sum", "mean"]

    if use_fp32_accum:
        # Convert to float32 for accumulation
        output_fp32 = output.to(torch.float32)
        src_fp32 = src.to(torch.float32)

        # Prepare tensors for scatter operation
        src_strided = src_fp32.as_strided(index.shape, src_fp32.stride())
        out_restrided = restride_dim(output_fp32, dim, index.shape)
        dim_size = output_fp32.size(dim)
        dim_stride = output_fp32.stride(dim)
        N = index.numel()

        int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
        use_int32_offset = all(map(int32_size_dim, (output_fp32, index, src_fp32)))

        # Call Triton kernel with float32
        _scatter_reduce_func(
            src_strided,
            index,
            out_restrided,
            output_fp32,
            dim_size,
            dim_stride,
            N,
            reduce,
            use_int32_offset,
        )

        # Convert back to float16
        output = output_fp32.to(torch.float16)
    else:
        # Use original dtype
        # Prepare tensors for scatter operation
        src_strided = src.as_strided(index.shape, src.stride())
        out_restrided = restride_dim(output, dim, index.shape)
        dim_size = output.size(dim)
        dim_stride = output.stride(dim)
        N = index.numel()

        int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
        use_int32_offset = all(map(int32_size_dim, (output, index, src)))

        # Call Triton kernel
        _scatter_reduce_func(
            src_strided,
            index,
            out_restrided,
            output,
            dim_size,
            dim_stride,
            N,
            reduce,
            use_int32_offset,
        )

    # For mean reduction, divide by count
    if reduce == "mean":
        # Count how many values were scattered to each position
        ones = torch.ones_like(src)

        # Use float32 for counting to match accumulation precision
        if use_fp32_accum:
            count_output = torch.zeros_like(output_fp32, dtype=torch.float32)
            ones_fp32 = ones.to(torch.float32)

            _scatter_reduce_func(
                ones_fp32.as_strided(index.shape, ones_fp32.stride()),
                index,
                restride_dim(output_fp32, dim, index.shape),
                count_output,
                dim_size,
                dim_stride,
                N,
                "sum",
                use_int32_offset,
            )

            if include_self:
                count_output = count_output + 1.0

            # Divide in float32, then convert back
            mask = count_output > 0
            count_output_safe = torch.where(
                mask, count_output, torch.ones_like(count_output)
            )
            output_fp32 = torch.where(
                mask, output_fp32 / count_output_safe, output_fp32
            )
            output = output_fp32.to(torch.float16)
        else:
            count_output = torch.zeros_like(output, dtype=torch.float32)

            _scatter_reduce_func(
                ones.as_strided(index.shape, ones.stride()),
                index,
                restride_dim(output, dim, index.shape),
                count_output,
                dim_size,
                dim_stride,
                N,
                "sum",
                use_int32_offset,
            )

            if include_self:
                count_output = count_output + 1.0

            # Avoid division by zero
            mask = count_output > 0
            output_float = output.to(torch.float32)
            count_output_safe = torch.where(
                mask, count_output, torch.ones_like(count_output)
            )
            output = torch.where(
                mask, (output_float / count_output_safe).to(output.dtype), output
            )

    return output


def scatter_reduce_(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    """In-place version of scatter_reduce."""
    logger.debug("GEMS SCATTER_REDUCE_")

    result = scatter_reduce(input, dim, index, src, reduce, include_self=include_self)
    input.copy_(result)
    return input
