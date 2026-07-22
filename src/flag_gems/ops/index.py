# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems import runtime
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


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _bucket(value, step):
    value = int(value)
    if value <= 0:
        return 0
    return ((value + step - 1) // step) * step


def _next_power_of_2(value):
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _shape_bucket(index_len, suffix_size):
    index_step = 32 if index_len < 262_144 else 131_072
    suffix_step = 32 if suffix_size < 256 else 128
    return (_bucket(index_len, index_step), _bucket(suffix_size, suffix_step))


def _index_candidate_configs(
    inp, start_dim, indices_len, tensor_indices, suffix_size=None
):
    """Select adjacent-index candidates from backend YAML configs."""

    index_len = int(tensor_indices[0].numel())
    if suffix_size is None:
        suffix_size = _volume(inp.shape[start_dim + indices_len :])

    max_tile_elems = 8192
    max_block_n = max(64, _bucket(suffix_size, 64) * 2)
    mid_block_n = max(64, _next_power_of_2(suffix_size))
    fallback_block_n = max(64, min(512, _bucket(suffix_size, 64)))
    max_configs = 2
    tuned_configs = runtime.get_tuned_config("index_adjacent")

    def valid(block_m, block_n):
        if index_len <= block_m // 2 and block_m > 4:
            return False
        if block_n > max_block_n:
            return False
        if block_m * block_n > max_tile_elems:
            return False
        if suffix_size != 1 and block_n == 1:
            return False
        if suffix_size <= block_n // 4 and block_n > 64:
            return False
        if suffix_size != 1 and indices_len > 1 and block_m > 16:
            return False
        return True

    def take(predicate, limit=max_configs):
        selected = []
        for config in tuned_configs:
            meta = config.kwargs
            block_m = int(meta["BLOCK_SIZE0"])
            block_n = int(meta["BLOCK_SIZE1"])
            if valid(block_m, block_n) and predicate(block_m, block_n):
                selected.append(config)
                if len(selected) >= limit:
                    break
        return selected

    if suffix_size == 1:
        if index_len <= 16:
            target_m = {8, 16}
        elif index_len <= 128:
            target_m = {16, 32}
        elif index_len <= 256:
            target_m = {32, 64}
        else:
            target_m = {64, 128}
        selected = take(lambda block_m, block_n: block_n == 1 and block_m in target_m)
        if selected:
            return selected

    if start_dim == 0 and indices_len == 1 and suffix_size >= 1024:
        selected = take(
            lambda block_m, block_n: block_m <= 4 and block_n >= 1024,
            limit=3,
        )
        if selected:
            return selected

    selected = []
    fallback = None
    for config in tuned_configs:
        meta = config.kwargs
        block_m = int(meta["BLOCK_SIZE0"])
        block_n = int(meta["BLOCK_SIZE1"])
        if not valid(block_m, block_n):
            continue
        if block_m == 4 and block_n == fallback_block_n:
            fallback = config
        if suffix_size <= 64:
            if block_m == 4 and block_n == fallback_block_n:
                return [config]
            continue
        if suffix_size < 256:
            if block_m > 4 and block_n == mid_block_n:
                selected.append(config)
        elif block_m > 4:
            selected.append(config)
        if len(selected) >= max_configs:
            return selected

    return selected or ([fallback] if fallback is not None else [])


def _write_index_configs(code: IndentedBuffer, configs):
    if configs is None:
        code.writeline('configs=runtime.get_tuned_config("index"),')
        return
    code.writeline("configs=[")
    with code.indent():
        for config in configs:
            code.writeline(
                "triton.Config("
                f"{config.kwargs!r}, "
                f"num_warps={config.num_warps}, "
                f"num_stages={config.num_stages}"
                "),"
            )
    code.writeline("],")


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")

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
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("pid1 = ext.program_id(axis=1)")
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


def generate_index_linearized_kernel(
    inp_shape,
    start_dim,
    indices_len,
    configs,
    kernel_name: str,
    code: IndentedBuffer,
):
    """Generate a kernel for contiguous input with adjacent tensor indices."""

    indexed_dims = [int(dim) for dim in inp_shape[start_dim : start_dim + indices_len]]
    indexed_volume = _volume(indexed_dims)
    indexed_strides = [
        _volume(indexed_dims[pos + 1 :]) for pos in range(len(indexed_dims))
    ]
    suffix_size = _volume(inp_shape[start_dim + indices_len :])

    code.writeline("@libentry()")
    code.writeline("@libtuner(")
    with code.indent():
        _write_index_configs(code, configs)
        code.writeline('key=["M"],')
        code.writeline('strategy=["align32"],')
        code.writeline("warmup=5,")
        code.writeline("rep=10,")
    code.writeline(")")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += [
            "out_ptr,",
            "M,",
            "N,",
            "BLOCK_SIZE0: tl.constexpr,",
            "BLOCK_SIZE1: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid_p = tl.program_id(axis=0)")
        code.writeline("pid_m = tl.program_id(axis=1)")
        code.writeline("pid_n = tl.program_id(axis=2)")
        code.newline()

        code.writeline(
            "index_offsets = pid_m * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)"
        )
        code.writeline(
            "suffix_offsets = pid_n * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)"
        )
        code.writeline("index_mask = index_offsets < M")
        code.newline()

        for i, dim in enumerate(indexed_dims):
            code.writeline(
                f"raw{i} = tl.load(indices{i}_ptr + index_offsets, mask=index_mask, other=0)"
            )
            code.writeline(f"idx{i} = tl.where(raw{i} < 0, raw{i} + {dim}, raw{i})")
        code.newline()

        linear_terms = [f"idx{i} * {indexed_strides[i]}" for i in range(indices_len)]
        valid_terms = [
            f"(idx{i} >= 0) & (idx{i} < {indexed_dims[i]})" for i in range(indices_len)
        ]
        code.writeline(f"linear_index = {' + '.join(linear_terms)}")
        code.writeline(f"valid_index = {' & '.join(valid_terms)}")
        code.newline()

        code.writeline("src_offsets = (")
        with code.indent():
            code.writeline(
                f"(pid_p * {indexed_volume} + linear_index[:, None]) * {suffix_size}"
            )
            code.writeline("+ suffix_offsets[None, :]")
        code.writeline(")")
        code.writeline("out_offsets = (")
        with code.indent():
            code.writeline(f"(pid_p * M + index_offsets[:, None]) * {suffix_size}")
            code.writeline("+ suffix_offsets[None, :]")
        code.writeline(")")
        code.writeline(
            "mask = index_mask[:, None] & valid_index[:, None] & "
            f"(suffix_offsets[None, :] < {suffix_size})"
        )
        code.newline()

        code.writeline("cur_value = tl.load(input_ptr + src_offsets, mask=mask)")
        code.writeline("tl.store(out_ptr + out_offsets, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_index_linearized_wrapper(
    inp_shape,
    start_dim,
    indices_len,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    prefix_size = _volume(inp_shape[:start_dim])
    suffix_size = _volume(inp_shape[start_dim + indices_len :])

    code.writeline(f"def {wrapper_name}(input, start_dim, indices, out):")
    with code.indent():
        code.writeline("M = indices[0].numel()")
        code.writeline(f"P = {prefix_size}")
        code.writeline(f"N = {suffix_size}")
        code.newline()
        # Match the kernel axes: prefix, flattened index elements, suffix.
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("P,")
            code.writeline("triton.cdiv(M, meta['BLOCK_SIZE0']),")
            code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']),")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["out,", "M,", "N,"]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_linearized_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    inp = inputs[0]
    start_dim = int(inputs[1])
    tensor_indices = inputs[2]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    configs = _index_candidate_configs(inp, start_dim, indices_len, tensor_indices)
    code = generate_imports(code)
    generate_index_linearized_kernel(
        tuple(inp.shape),
        start_dim,
        indices_len,
        configs,
        kernel_name,
        code,
    )
    generate_index_linearized_wrapper(
        tuple(inp.shape),
        start_dim,
        indices_len,
        wrapper_name,
        kernel_name,
        code,
    )
    return code


class LinearizedAdjacentIndexFunction:
    """Code cache for contiguous input with adjacent tensor indices."""

    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, start_dim, tensor_indices, out = args
        full_args = (inp, start_dim, tensor_indices)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_linearized_code(
                full_args,
                "_index_linearized_wrapper",
                "_index_linearized_jit_function",
                code,
            )

            file_name = f"index_linearized_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_index_linearized_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_linearized_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, start_dim, tensor_indices = args[0], int(args[1]), args[2]
        inp_shape = "_".join(str(int(dim)) for dim in inp.shape)
        indices_len = len(tensor_indices)
        index_len = int(tensor_indices[0].numel()) if tensor_indices else 0
        suffix_size = _volume(inp.shape[start_dim + indices_len :])
        index_bucket, suffix_bucket = _shape_bucket(index_len, suffix_size)
        return (
            f"inp_shape_{inp_shape}_start_dim_{start_dim}_"
            f"indices_len_{indices_len}_"
            f"dtype_{str(inp.dtype).replace('.', '_')}_"
            f"m_{index_bucket}_n_{suffix_bucket}"
        )


_linearized_adjacent_index_func = LinearizedAdjacentIndexFunction()


def _tensor_index_dims(indices):
    return [dim for dim, index in enumerate(indices) if index is not None]


def _are_adjacent_tensor_indices(tensor_index_dims):
    if not tensor_index_dims:
        return False
    return tensor_index_dims == list(
        range(tensor_index_dims[0], tensor_index_dims[0] + len(tensor_index_dims))
    )


def _linearized_adjacent_index_configs(inp, indices):
    """Return configs for the adjacent-index path, or [] when not enabled.

    Backend-local `index_adjacent` configs are preferred. When they are absent,
    FlagGems follows its default tuning-config fallback.
    """

    tensor_index_dims = _tensor_index_dims(indices)
    if (
        not tensor_index_dims
        or not inp.is_contiguous()
        or not _are_adjacent_tensor_indices(tensor_index_dims)
    ):
        return []

    start_dim = tensor_index_dims[0]
    tensor_indices = [indices[dim] for dim in tensor_index_dims]
    return _index_candidate_configs(inp, start_dim, len(tensor_indices), tensor_indices)


def _run_linearized_adjacent_index(inp, indices):
    tensor_index_dims = _tensor_index_dims(indices)
    start_dim = tensor_index_dims[0]
    tensor_indices = [indices[dim] for dim in tensor_index_dims]
    # Preserve PyTorch advanced-indexing output order without moving indexed
    # dimensions to the front, e.g. x[:, idx, :] and x[:, idx0, idx1, :].
    out_shape = (
        list(inp.shape[:start_dim])
        + list(tensor_indices[0].shape)
        + list(inp.shape[start_dim + len(tensor_index_dims) :])
    )
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0 or out.numel() == 0:
        return out.contiguous()

    # The generated kernel assumes contiguous index tensors and contiguous input.
    tensor_indices = [idx.contiguous() for idx in tensor_indices]
    _linearized_adjacent_index_func(inp, start_dim, tensor_indices, out)
    return out.contiguous()


def index(inp, indices):
    logger.debug("GEMS INDEX")
    original_indices = list(indices)  # Save original indices for later checks
    indices = list(indices)

    if not indices:
        raise ValueError("at least one index must be provided")

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]

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
                    raise IndexError(
                        f"too many indices for tensor of dimension {inp.ndim}"
                    )
                # Check shape matches
                for j in range(index.ndim):
                    if index.shape[j] != inp.shape[k + j]:
                        raise IndexError(
                            f"The shape of the mask {index.shape} at index {i} "
                            f"does not match the shape of the indexed tensor "
                            f"{inp.shape} at index {k + j}"
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

    # Save for later use
    has_any_tensor = any(idx is not None for idx in indices)
    starts_with_none = indices[0] is None if indices else False

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

    # Adjacent tensor indices on contiguous input, e.g. x[idx, :] or
    # x[:, idx0, idx1, :], can be linearized in place when the resolved
    # backend/default config pool contains a shape-compatible candidate.
    adjacent_index_configs = (
        _linearized_adjacent_index_configs(inp, indices)
        if has_contiguous_subspace
        else []
    )
    if adjacent_index_configs:
        return _run_linearized_adjacent_index(inp, indices)

    # Transpose if not contiguous OR starts with None (and has tensor indices)
    need_post_process = False
    first_tensor_dim = None
    if not has_contiguous_subspace or (starts_with_none and has_any_tensor):
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

        # Check if we need post-processing
        # (only when originally started with None and was contiguous)
        if starts_with_none and has_any_tensor and has_contiguous_subspace:
            need_post_process = True
            # Find first tensor dimension in original indices
            for i, idx in enumerate(original_indices):
                if idx is not None:
                    first_tensor_dim = i
                    break

    # Step 5: Now indices have contiguous subspace (after potential transpose)
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

    # Step 6: Build output shape and create output tensor
    out_shape = before_shape + replacement_shape + after_shape
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Step 7: Handle empty tensor case
    if inp.numel() == 0:
        return out.contiguous()

    # Step 8: Extract only tensor indices for kernel
    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        # All None, just reshape
        return inp.view(*out_shape).contiguous()

    # Step 9: Call kernel with tensor indices
    _index_func(inp, tensor_indices, out)

    # Step 10: Post-process if needed (for originally contiguous tensor indices starting with None)
    if need_post_process:
        # Calculate index_rank from the first tensor index
        index_rank = tensor_indices[0].ndim
        # Create permutation order to move broadcast dimensions to correct position
        pre_dims = list(range(index_rank, index_rank + first_tensor_dim))
        broadcast_dims = list(range(index_rank))
        post_dims = list(range(index_rank + first_tensor_dim, out.ndim))
        new_order = pre_dims + broadcast_dims + post_dims
        out = out.permute(new_order)

    return out.contiguous()
