import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.ops.gather import gather
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def get_max_rank_shape(indices: List[torch.Tensor]) -> List[int]:
    # Filter out None values (basic indexing markers)
    tensor_indices = [idx for idx in indices if idx is not None]
    if len(tensor_indices) == 0:
        return []
    max_rank = max([len(index.shape) for index in tensor_indices])
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in tensor_indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0:
                max_num = max(max_num, index.shape[axis])  #
        shape[max_rank - 1 - i] = max_num
    return shape


def broadcast_indices(indices, target_shape):
    for i, index in enumerate(indices):
        if index is not None and tuple(index.shape) != tuple(target_shape):
            indices[i] = torch.broadcast_to(index, target_shape)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")

    code.newline()
    code.newline()
    return code


def generate_index_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline("@libtuner(")
    with code.indent():
        code.writeline('configs=runtime.get_tuned_config("index"),')
        code.writeline('key=["M", "N"],')
        code.writeline('restore_value=["input_ptr"],')
        code.writeline('strategy=["align32", "align32"],')
        code.writeline("warmup=5,")
        code.writeline("rep=10,")
    code.writeline(")")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["out_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [f"out_stride{i}," for i in range(index_rank + inp_rank - indices_len)]
        args += [
            "M,",
            "N,",
            "BLOCK_SIZE0: tl.constexpr,",
            "BLOCK_SIZE1: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = tle.program_id(axis=0)")
        code.writeline("pid1 = tle.program_id(axis=1)")
        code.writeline(
            "offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]"
        )
        if inp_rank == indices_len:
            code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
        else:
            code.writeline(
                "offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]"
            )
        code.newline()
        code.writeline("cur_idx = offset0")
        for i in range(index_rank - 1, -1, -1):
            code.writeline(f"indices_idx{i} = cur_idx % indices0_shape{i}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{i}")
        code.newline()
        code.writeline("cur_idx = offset1")
        for i in range(inp_rank - 1, indices_len - 1, -1):
            code.writeline(f"input_idx{i} = cur_idx % input_shape{i}")
            code.writeline(f"cur_idx = cur_idx // input_shape{i}")
        code.newline()
        code.writeline("mask0 = offset0 < M")
        for i in range(indices_len):
            comp = [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
            code.writeline(
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0)"
            )
        code.newline()
        index_mask = [
            f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
            for i in range(indices_len)
        ]
        code.writeline(f"index_mask = {' & '.join(index_mask)}")
        code.writeline("mask1 = offset1 < N")
        code.writeline("mask = index_mask & mask0 & mask1")
        code.newline()
        comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
        comp += [
            f"input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)
        ]
        code.writeline(f"input_offset = {' + '.join(comp)}")
        comp = [f"indices_idx{i} * out_stride{i}" for i in range(index_rank)]
        comp += [
            f"input_idx{indices_len + i} * out_stride{index_rank + i}"
            for i in range(inp_rank - indices_len)
        ]
        code.writeline(f"out_offset = {' + '.join(comp)}")
        code.newline()
        code.writeline("cur_value = tl.load(input_ptr + input_offset , mask = mask)")
        code.writeline("tl.store(out_ptr + out_offset, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_index_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code.writeline(f"def {wrapper_name}(input, indices, out):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("out_shape = out.shape")
        code.writeline("out_stride = out.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(M, meta['BLOCK_SIZE0']), ")
            code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']), ")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["out,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"out_stride[{i}]," for i in range(index_rank + inp_rank - indices_len)
            ]
            args += ["M,", "N,"]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    inp_rank = inputs[0].ndim
    # Filter out None values to get actual tensor indices
    tensor_indices = [idx for idx in inputs[1] if idx is not None]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = generate_imports(code)
    generate_index_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_index_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, out = args
        full_args = (inp, tensor_indices)
        
        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
                "_index_wrapper",
                "_index_jit_function",
                code,
            )

            file_name = f"index_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices = args[0], args[1]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        if indices_len == 0:
            index_rank = 0
        else:
            index_rank = tensor_indices[0].ndim
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_func = IndexFunction()


def index(inp, indices):
    logger.debug("GEMS INDEX")
    indices = list(indices)
    
    if not indices:
        raise ValueError("at least one index must be provided")
    
    # Step 1: Process indices (convert bool/int8 to long, handle None)
    # Following PyTorch meta implementation
    processed_indices = []
    for i, index in enumerate(indices):
        if index is not None:
            # Check dtype
            if index.dtype in [torch.int8, torch.bool]:
                # Convert boolean/int8 mask to long indices
                nonzero = index.nonzero()
                k = len(processed_indices)
                if k + index.ndim > inp.ndim:
                    raise IndexError(f"too many indices for tensor of dimension {inp.ndim}")
                # Check shape matches
                for j in range(index.ndim):
                    if index.shape[j] != inp.shape[k + j]:
                        raise IndexError(
                            f"The shape of the mask {index.shape} at index {i} "
                            f"does not match the shape of the indexed tensor {inp.shape} at index {k + j}"
                        )
                # Extract indices from nonzero
                for j in range(index.ndim):
                    processed_indices.append(nonzero.select(1, j))
            elif index.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                processed_indices.append(index)
            else:
                raise TypeError(
                    "tensors used as indices must be long, int, byte or bool tensors"
                )
        else:
            processed_indices.append(None)
    
    indices = processed_indices
    
    # Check indices count
    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )
    
    # Step 2: Broadcast indices (only tensor indices, not None)
    tensor_indices = [idx for idx in indices if idx is not None]
    if tensor_indices:
        # Broadcast all tensor indices together
        if len(tensor_indices) > 1:
            tensor_indices = list(torch.broadcast_tensors(*tensor_indices))
        # Update indices list with broadcasted tensors
        tensor_idx = 0
        for i in range(len(indices)):
            if indices[i] is not None:
                indices[i] = tensor_indices[tensor_idx]
                tensor_idx += 1
    
    # Step 3: Add missing None indices (pad to input.ndim)
    while len(indices) < inp.ndim:
        indices.append(None)
    
    # Step 4: Check if has contiguous subspace
    # (all non-None tensors are adjacent)
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True
    
    # Step 5: Transpose to front if needed
    # If not contiguous, transpose input so all non-None indices come first
    if not has_contiguous_subspace:
        dims = []
        transposed_indices = []
        # First add all non-None index positions
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        # Then add all None positions
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        # Permute input
        inp = inp.permute(dims)
        indices = transposed_indices
    
    # Step 6: Now indices have contiguous subspace
    # Calculate output shape: before_shape + replacement_shape + after_shape
    before_shape = []
    after_shape = []
    replacement_shape = []
    
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                # None after tensor indices -> goes to after_shape
                after_shape.append(inp.shape[dim])
            else:
                # None before tensor indices -> goes to before_shape
                before_shape.append(inp.shape[dim])
        else:
            # First tensor index determines replacement_shape
            if not replacement_shape:
                replacement_shape = list(index.shape)
    
    # Step 7: Build output shape and create output tensor
    out_shape = before_shape + replacement_shape + after_shape
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    
    # Step 8: Handle empty tensor case
    if inp.numel() == 0:
        return out
    
    # Step 9: Extract only tensor indices for kernel
    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        # All None, just reshape
        return inp.view(*out_shape)
    
    # Step 10: Call kernel with tensor indices
    # Note: kernel needs to handle the fact that input was potentially permuted
    # and output shape includes None dimensions
    if inp.ndim == 1 and len(tensor_indices) == 1:
        return gather(inp, 0, tensor_indices[0])
    
    # For mixed indexing, we need to adjust the kernel call
    # The kernel should work with the permuted input and handle output shape correctly
    _index_func(inp, tensor_indices, out)
    return out
