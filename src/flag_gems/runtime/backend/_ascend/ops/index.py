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

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_LAYOUT_2D_TILE = 0
_LAYOUT_SUFFIX_LOOP = 1


@triton.jit
def index_kernel_func(
    input_ptr,
    stride: tl.constexpr,
    index_len,
    index_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DATA_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)

    for i in range(0, BLOCK_SIZE):
        offset = pid0 * BLOCK_SIZE + i

        if offset < index_len:
            in_start_index = tl.load(index_ptr + offset) * stride
            out_start_offset = offset * stride
            loop_num = (stride - 1) // MAX_DATA_SIZE + 1

            for loop_idx in range(0, loop_num):
                inner_offset = loop_idx * MAX_DATA_SIZE + tl.arange(0, MAX_DATA_SIZE)
                mask = inner_offset < stride
                cur_value = tl.load(
                    input_ptr + in_start_index + inner_offset, mask=mask
                )
                tl.store(
                    out_ptr + out_start_offset + inner_offset, cur_value, mask=mask
                )


def index_wrapper(input, indices, out):
    """
    Simple kernel wrapper for contiguous tensor indices starting from dim 0
    """
    input_shape = input.shape
    input_dim = len(input_shape)
    indices_dim = len(indices)
    stride = 1

    for i in range(indices_dim, input_dim):
        stride *= input_shape[i]

    index_len = indices[0].numel()
    if index_len <= 0:
        return

    actual_index = indices[0]
    for idx in range(0, indices_dim - 1):
        actual_index = actual_index * input_shape[idx + 1] + indices[idx + 1]

    BLOCK_SIZE = 32
    MAX_DATA_SIZE = 8 * 1024

    grid = lambda meta: (triton.cdiv(index_len, meta["BLOCK_SIZE"]),)

    index_kernel_func[grid](
        input,
        stride,
        index_len,
        actual_index,
        out,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_DATA_SIZE=MAX_DATA_SIZE,
    )


def _shape_product(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def _bucket(value, step):
    value = int(value)
    if value <= 0:
        return 0
    return ((value + step - 1) // step) * step


def _shape_bucket(index_len, suffix_size):
    index_step = 32 if index_len < 262_144 else 131_072
    suffix_step = 32 if suffix_size < 256 else 128
    return (_bucket(index_len, index_step), _bucket(suffix_size, suffix_step))


def _small_suffix_block_s(suffix_size):
    block_s = 1
    while block_s * 2 <= suffix_size and block_s * 2 <= 64:
        if suffix_size % (block_s * 2) != 0:
            break
        block_s *= 2
    return block_s


def _adjacent_index_config(block_m, block_s, layout, suffix_tiles_per_program):
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_S": block_s,
            "LAYOUT": layout,
            "SUFFIX_TILES_PER_PROGRAM": suffix_tiles_per_program,
        },
        num_warps=4,
        num_stages=2,
    )


def _find_adjacent_index_config(
    configs,
    block_m,
    block_s,
    layout,
    suffix_tiles_per_program=1,
):
    for config in configs:
        meta = config.kwargs
        if (
            int(meta["BLOCK_M"]) == block_m
            and int(meta["BLOCK_S"]) == block_s
            and int(meta["LAYOUT"]) == layout
            and int(meta["SUFFIX_TILES_PER_PROGRAM"]) == suffix_tiles_per_program
        ):
            return config
    return None


def _adjacent_index_candidate_configs(index_len, suffix_size, indices_len, prefix_size):
    """Select Ascend adjacent-index candidates from backend YAML configs."""

    max_grid_axis = 65535
    max_tile_elems = 4096 if suffix_size >= 256 else 8192

    if suffix_size < 64:
        block_s = _small_suffix_block_s(suffix_size)
        block_m = 16 if indices_len > 1 else 32
        raw = ((block_m, block_s, _LAYOUT_2D_TILE),)
    elif suffix_size == 64:
        raw = ((4, 64, _LAYOUT_2D_TILE),)
    elif suffix_size < 256:
        block_s = 64 if suffix_size <= 64 else 128
        raw = (
            (16, block_s, _LAYOUT_2D_TILE),
            (8 if indices_len > 1 else 32, block_s, _LAYOUT_2D_TILE),
        )
    elif indices_len > 1:
        raw = (
            (8, 512, _LAYOUT_2D_TILE),
            (16, 256, _LAYOUT_2D_TILE),
        )
    else:
        raw = (
            (8, 512, _LAYOUT_2D_TILE),
            (16, 256, _LAYOUT_2D_TILE),
        )

    min_block_m = max(1, _ceil_div(index_len, max_grid_axis))
    use_suffix_loop = _should_use_adjacent_suffix_loop(
        index_len, suffix_size, prefix_size
    )
    max_configs = 2
    configs = []
    seen = set()
    tuned_configs = runtime.get_tuned_config("index_adjacent_ascend")

    for block_m, block_s, layout in raw:
        if block_m < min_block_m:
            continue
        if block_m * block_s > max_tile_elems:
            continue
        if _ceil_div(index_len, block_m) > max_grid_axis:
            continue
        if _ceil_div(suffix_size, block_s) > max_grid_axis:
            continue
        if indices_len > 1 and block_m > 16:
            continue
        key = (layout, block_m, block_s, 1)
        if key in seen:
            continue
        config = _find_adjacent_index_config(tuned_configs, block_m, block_s, layout, 1)
        if config is None:
            config = _adjacent_index_config(block_m, block_s, layout, 1)
        seen.add(key)
        configs.append(config)
        if len(configs) >= max_configs:
            break

    if use_suffix_loop:
        block_m, block_s, suffix_tiles_per_program = _adjacent_suffix_loop_config(
            index_len, suffix_size, prefix_size
        )
        config = _find_adjacent_index_config(
            tuned_configs,
            block_m,
            block_s,
            _LAYOUT_SUFFIX_LOOP,
            suffix_tiles_per_program,
        )
        if config is None:
            config = _adjacent_index_config(
                block_m, block_s, _LAYOUT_SUFFIX_LOOP, suffix_tiles_per_program
            )
        configs.append(config)

    if configs:
        return configs

    block_m = max(16, min(256, min_block_m))
    block_s = max(1, min(int(suffix_size), max_tile_elems // block_m))
    config = _find_adjacent_index_config(
        tuned_configs, block_m, block_s, _LAYOUT_2D_TILE, 1
    )
    return [config or _adjacent_index_config(block_m, block_s, _LAYOUT_2D_TILE, 1)]


def _adjacent_suffix_loop_config(index_len, suffix_size, prefix_size):
    block_m = 32
    block_s = 8 * 1024
    index_blocks = _ceil_div(index_len, block_m)
    suffix_tiles = _ceil_div(suffix_size, block_s)
    base_programs = int(prefix_size) * index_blocks

    if base_programs >= 8:
        suffix_groups = 1
    elif base_programs >= 4:
        suffix_groups = 2
    else:
        suffix_groups = 4

    suffix_groups = min(suffix_tiles, suffix_groups)
    return block_m, block_s, _ceil_div(suffix_tiles, suffix_groups)


def _should_use_adjacent_suffix_loop(index_len, suffix_size, prefix_size):
    block_m = 32
    block_s = 8 * 1024
    index_blocks = _ceil_div(index_len, block_m)
    suffix_tiles = _ceil_div(suffix_size, block_s)
    base_programs = int(prefix_size) * index_blocks

    return suffix_tiles >= 8 and base_programs <= 16


def _write_adjacent_index_configs(code, configs):
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


def _write_adjacent_index_code(
    inp_shape,
    start_dim,
    indices_len,
    configs,
    wrapper_name,
    kernel_name,
):
    code = IndentedBuffer()
    indexed_dims = [int(dim) for dim in inp_shape[start_dim : start_dim + indices_len]]
    indexed_volume = _shape_product(indexed_dims)
    indexed_strides = [
        _shape_product(indexed_dims[pos + 1 :]) for pos in range(len(indexed_dims))
    ]
    prefix_size = _shape_product(inp_shape[:start_dim])
    suffix_size = _shape_product(inp_shape[start_dim + indices_len :])

    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.newline()
    code.writeline("@libentry()")
    code.writeline("@libtuner(")
    with code.indent():
        _write_adjacent_index_configs(code, configs)
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
        args += [
            "out_ptr,",
            "M,",
            "N,",
            "BLOCK_M: tl.constexpr,",
            "BLOCK_S: tl.constexpr,",
            "LAYOUT: tl.constexpr,",
            "SUFFIX_TILES_PER_PROGRAM: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid_p = tl.program_id(axis=0)")
        code.writeline("pid_m = tl.program_id(axis=1)")
        code.writeline("pid_s = tl.program_id(axis=2)")
        code.newline()
        # Layout 0: one program owns a BLOCK_M x BLOCK_S tile.
        # Layout 1: one program loops over grouped suffix tiles.
        code.writeline(f"if LAYOUT == {_LAYOUT_2D_TILE}:")
        with code.indent():
            code.writeline("index_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
            code.writeline("suffix_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)")
            code.writeline("index_mask = index_offsets < M")
            code.newline()
            for i, dim in enumerate(indexed_dims):
                code.writeline(
                    f"raw{i} = tl.load(indices{i}_ptr + index_offsets, mask=index_mask, other=0)"
                )
                code.writeline(f"idx{i} = tl.where(raw{i} < 0, raw{i} + {dim}, raw{i})")
            code.newline()
            linear_terms = [
                f"idx{i} * {indexed_strides[i]}" for i in range(indices_len)
            ]
            valid_terms = [
                f"(idx{i} >= 0) & (idx{i} < {indexed_dims[i]})"
                for i in range(indices_len)
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
            code.writeline("cur_value = tl.load(input_ptr + src_offsets, mask=mask)")
            code.writeline("tl.store(out_ptr + out_offsets, cur_value, mask=mask)")
        code.writeline("else:")
        with code.indent():
            code.writeline("for index_i in range(0, BLOCK_M):")
            with code.indent():
                code.writeline("index_offset = pid_m * BLOCK_M + index_i")
                code.writeline("index_mask = index_offset < M")
                for i, dim in enumerate(indexed_dims):
                    code.writeline(
                        f"raw{i} = tl.load(indices{i}_ptr + index_offset, mask=index_mask, other=0)"
                    )
                    code.writeline(
                        f"idx{i} = tl.where(raw{i} < 0, raw{i} + {dim}, raw{i})"
                    )
                code.newline()
                linear_terms = [
                    f"idx{i} * {indexed_strides[i]}" for i in range(indices_len)
                ]
                valid_terms = [
                    f"(idx{i} >= 0) & (idx{i} < {indexed_dims[i]})"
                    for i in range(indices_len)
                ]
                code.writeline(f"linear_index = {' + '.join(linear_terms)}")
                code.writeline(f"valid_index = {' & '.join(valid_terms)}")
                code.newline()
                code.writeline("for suffix_tile in range(0, SUFFIX_TILES_PER_PROGRAM):")
                with code.indent():
                    code.writeline(
                        "suffix_offsets = "
                        "(pid_s * SUFFIX_TILES_PER_PROGRAM + suffix_tile) "
                        "* BLOCK_S + tl.arange(0, BLOCK_S)"
                    )
                    code.writeline("src_offsets = (")
                    with code.indent():
                        code.writeline(
                            f"(pid_p * {indexed_volume} + linear_index) * {suffix_size}"
                        )
                        code.writeline("+ suffix_offsets")
                    code.writeline(")")
                    code.writeline("out_offsets = (")
                    with code.indent():
                        code.writeline(f"(pid_p * M + index_offset) * {suffix_size}")
                        code.writeline("+ suffix_offsets")
                    code.writeline(")")
                    code.writeline(
                        "mask = index_mask & valid_index & "
                        f"(suffix_offsets < {suffix_size})"
                    )
                    code.writeline(
                        "cur_value = tl.load(input_ptr + src_offsets, mask=mask)"
                    )
                    code.writeline(
                        "tl.store(out_ptr + out_offsets, cur_value, mask=mask)"
                    )

    code.newline()
    code.writeline(f"def {wrapper_name}(input, start_dim, indices, out):")
    with code.indent():
        code.writeline("M = indices[0].numel()")
        code.writeline(f"P = {prefix_size}")
        code.writeline(f"N = {suffix_size}")
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("P,")
            code.writeline("triton.cdiv(M, meta['BLOCK_M']),")
            code.writeline(
                "triton.cdiv(triton.cdiv(N, meta['BLOCK_S']), meta['SUFFIX_TILES_PER_PROGRAM'])"
            )
            code.writeline(f"if meta['LAYOUT'] == {_LAYOUT_SUFFIX_LOOP}")
            code.writeline("else triton.cdiv(N, meta['BLOCK_S']),")
        code.writeline(")")
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += [
                "out,",
                "M,",
                "N,",
            ]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    return code


class AscendAdjacentIndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads = {}

    def __call__(self, inp, start_dim, tensor_indices, out):
        key = self.arg_key(inp, start_dim, tensor_indices)
        if key not in self.overloads:
            indices_len = len(tensor_indices)
            index_len = int(tensor_indices[0].numel())
            prefix_size = _shape_product(inp.shape[:start_dim])
            suffix_size = _shape_product(inp.shape[start_dim + indices_len :])
            configs = _adjacent_index_candidate_configs(
                index_len, suffix_size, indices_len, prefix_size
            )
            code = _write_adjacent_index_code(
                tuple(inp.shape),
                start_dim,
                indices_len,
                configs,
                "_index_adjacent_wrapper",
                "_index_adjacent_kernel",
            )
            file_name = f"ascend_index_adjacent_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_ascend_index_adjacent_{key}", file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.overloads[key] = getattr(module, "_index_adjacent_wrapper")
        return self.overloads[key](inp, start_dim, tensor_indices, out)

    def arg_key(self, inp, start_dim, tensor_indices):
        inp_shape = "_".join(str(int(dim)) for dim in inp.shape)
        indices_len = len(tensor_indices)
        index_len = int(tensor_indices[0].numel()) if tensor_indices else 0
        suffix_size = _shape_product(inp.shape[start_dim + indices_len :])
        index_bucket, suffix_bucket = _shape_bucket(index_len, suffix_size)
        return (
            f"shape_{inp_shape}_start_{int(start_dim)}_indices_{indices_len}_"
            f"dtype_{str(inp.dtype).replace('.', '_')}_m_{index_bucket}_n_{suffix_bucket}"
        )


_adjacent_index_func = AscendAdjacentIndexFunction()


def _tensor_index_dims(indices):
    return [dim for dim, index in enumerate(indices) if index is not None]


def _tensor_index_dims_are_adjacent(tensor_index_dims):
    if not tensor_index_dims:
        return False
    return tensor_index_dims == list(
        range(tensor_index_dims[0], tensor_index_dims[0] + len(tensor_index_dims))
    )


def _can_use_linearized_adjacent_index(inp, indices):
    # Keep this condition aligned with flag_gems.ops.index.index.
    tensor_index_dims = _tensor_index_dims(indices)
    # The Ascend specialized kernel is currently validated for floating payloads;
    # integer tensors keep the upstream redispatch path for exact semantics.
    supported_dtype = inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    return (
        supported_dtype
        and tensor_index_dims
        and inp.is_contiguous()
        and _tensor_index_dims_are_adjacent(tensor_index_dims)
    )


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

    tensor_indices = [idx.contiguous() for idx in tensor_indices]
    _adjacent_index_func(inp, start_dim, tensor_indices, out)
    return out.contiguous()


def _aten_index_fallback(inp, indices):
    # Redispatch bypasses the Python registration installed by only_enable/use_gems;
    # using inp[...] here would re-enter this implementation for fallback cases.
    aten_indices = [idx if idx is not None else None for idx in indices]
    return torch.ops.aten.index.Tensor.redispatch(_FALLBACK_KEYSET, inp, aten_indices)


def index(inp, indices):
    logger.debug("GEMS_ASCEND INDEX")
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
    starts_from_zero = False
    for i, index in enumerate(indices):
        if state == 0:
            if index is not None:
                if i == 0:
                    starts_from_zero = True
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
    # x[:, idx0, idx1, :], can be linearized in place. This preserves
    # PyTorch's output order and avoids the transpose-to-front path below.
    if has_contiguous_subspace and _can_use_linearized_adjacent_index(inp, indices):
        return _run_linearized_adjacent_index(inp, indices)

    # Step 5: Transpose-to-front/complex indexing fallback.
    if not has_contiguous_subspace or not starts_from_zero:
        return _aten_index_fallback(inp, indices)

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
    if inp.numel() == 0 or out.numel() == 0:
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
        return torch.gather(inp, 0, tensor_indices[0])

    index_wrapper(inp, tensor_indices, out)
    return out
