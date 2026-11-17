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

import importlib.util
import logging
import os
from functools import lru_cache
from hashlib import sha256

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import broadcast_shapes

logger = logging.getLogger(__name__)

_LAYOUT_2D_TILE = 0
_LAYOUT_SUFFIX_LOOP = 1


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
    tensor_index_dims = _tensor_index_dims(indices)
    supported_dtype = inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not (
        supported_dtype
        and tensor_index_dims
        and _tensor_index_dims_are_adjacent(tensor_index_dims)
    ):
        return False
    if inp.is_contiguous():
        return True
    output_numel = indices[tensor_index_dims[0]].numel() * _shape_product(
        inp.size(dim) for dim in range(inp.ndim) if dim not in tensor_index_dims
    )
    # A direct gather is slower for small views. Bound layout-copy work by
    # either a small absolute budget or a constant multiple of output size.
    return (
        inp.numel() * inp.element_size() <= 128 * 1024**2
        or inp.numel() <= 4 * output_numel
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

    inp = inp.contiguous()
    tensor_indices = [idx.contiguous() for idx in tensor_indices]
    _adjacent_index_func(inp, start_dim, tensor_indices, out)
    return out.contiguous()


@libentry()
@triton.jit
def _index_check_bounds_kernel(
    index, valid, N, SIZE: tl.constexpr, DEVICE_CHECK: tl.constexpr, BLOCK: tl.constexpr
):
    good = tl.full((), True, tl.int1)
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        raw = tl.load(index + offsets, offsets < N, other=0).to(tl.int64)
        good &= (
            tl.min(((offsets >= N) | ((raw >= -SIZE) & (raw < SIZE))).to(tl.int32), 0)
            != 0
        )
    if DEVICE_CHECK:
        if not good:
            tl.device_assert(False, "index out of bounds")
    else:
        tl.store(valid + tl.program_id(0), good)


def _check_index_bounds(index, size):
    if index.numel() == 0:
        return
    index = index.contiguous()
    block = min(1024, triton.next_power_of_2(index.numel()))
    programs = min(48, triton.cdiv(index.numel(), block))
    capturing = torch_device_fn.is_current_stream_capturing()
    valid = (
        None
        if capturing
        else torch.empty(programs, device=index.device, dtype=torch.int32)
    )
    _index_check_bounds_kernel[(programs,)](
        index, valid, index.numel(), size, capturing, block, debug=capturing
    )
    # Capture cannot read a host scalar. Enable the device assertion explicitly
    # on that path; eager calls keep the normal Python IndexError contract.
    if not capturing and not all(valid.cpu().tolist()):
        raise IndexError(f"index is out of bounds for dimension with size {size}")


@triton.jit
def _mask_block_counts(mask, counts, N, BLOCK: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        selected = tl.load(mask + offsets, offsets < N, other=0) != 0
        tl.store(counts + tile, tl.sum(selected.to(tl.int32), 0))


@triton.jit
def _mask_prefix_counts(counts, prefixes, total, TILES, BLOCK: tl.constexpr):
    carry = tl.full((), 0, tl.int64)
    for start in range(0, TILES, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        values = tl.load(counts + offsets, offsets < TILES, other=0)
        inclusive = tl.cumsum(values, 0)
        tl.store(prefixes + offsets, carry + inclusive - values, offsets < TILES)
        carry += tl.sum(values, 0).to(tl.int64)
    tl.store(total, carry)


@triton.jit
def _mask_scatter_coordinates(
    mask, prefixes, out, N, COUNT, SHAPE: tl.constexpr, BLOCK: tl.constexpr
):
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        selected = (offsets < N) & (tl.load(mask + offsets, offsets < N, other=0) != 0)
        ranks = tl.cumsum(selected.to(tl.int32), 0) - 1
        prefix = tl.load(prefixes + tile)
        destination = tl.where(selected, prefix + ranks, 0)
        remaining = offsets
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE[dim]
            remaining = remaining // SHAPE[dim]
            tl.store(out + dim * COUNT + destination, coordinate, selected)


def _boolean_coordinates(mask, device):
    # Dynamic output sizes require a host count, as with native nonzero.
    # Reject capture before launching any device work rather than retrying later.
    if torch_device_fn.is_current_stream_capturing():
        raise RuntimeError(
            "boolean indexing with a data-dependent output shape cannot be captured"
        )
    mask = mask.to(device).contiguous()
    if mask.numel() == 0:
        return [
            torch.empty(0, device=mask.device, dtype=torch.int64) for _ in mask.shape
        ]
    tiles = triton.cdiv(mask.numel(), 1024)
    counts = torch.empty(tiles, device=mask.device, dtype=torch.int32)
    prefixes = torch.empty(tiles, device=mask.device, dtype=torch.int64)
    total = torch.empty((), device=mask.device, dtype=torch.int64)
    grid = (min(tiles, 65535),)
    _mask_block_counts[grid](mask, counts, mask.numel(), BLOCK=1024)
    _mask_prefix_counts[(1,)](counts, prefixes, total, tiles, BLOCK=1024)
    count = int(total.item())
    coordinates = torch.empty((mask.ndim, count), device=mask.device, dtype=torch.int64)
    if count:
        _mask_scatter_coordinates[grid](
            mask,
            prefixes,
            coordinates,
            mask.numel(),
            count,
            tuple(mask.shape),
            BLOCK=1024,
        )
    return [coordinates.select(0, dim) for dim in range(mask.ndim)]


@lru_cache(maxsize=256)
def _gather_function(
    shape,
    strides,
    index_dims,
    index_strides,
    broadcast_shape,
    output_shape,
    plain_axes,
    broadcast_axes,
):
    # Generate only shape/stride address arithmetic; payloads and index values
    # remain device inputs. No contiguous copies or native index redispatch.
    lines = [
        "import triton",
        "import triton.language as tl",
        "from flag_gems.utils import libentry",
        "@libentry()",
        "@triton.jit",
    ]
    args = (
        ["inp"]
        + [f"index{i}" for i in range(len(index_dims))]
        + ["out", "N", "BLOCK: tl.constexpr", "PROGRAMS: tl.constexpr"]
    )
    lines += [
        "def kernel(" + ", ".join(args) + "):",
        "    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), PROGRAMS):",
        "        linear = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "        active = linear < N",
        "        remaining = linear.to(tl.int64)",
        "        source = tl.full((BLOCK,), 0, tl.int64)",
    ]
    for axis in reversed(range(len(output_shape))):
        lines += [
            f"        c{axis} = remaining % {output_shape[axis]}",
            f"        remaining = remaining // {output_shape[axis]}",
        ]
    for dim, axis in plain_axes:
        lines += [f"        source += c{axis} * {strides[dim]}"]
    for i, dim in enumerate(index_dims):
        # Extent-one coordinates are always zero. Leaving their strides in
        # the expression creates a splatted pointer under a vector mask after
        # folding, which FlagTree cannot lower. Use the scalar load below.
        terms = [
            f"c{axis} * {stride}"
            for axis, stride in zip(broadcast_axes, index_strides[i])
            if stride and output_shape[axis] > 1
        ]
        offset = " + ".join(terms)
        load = (
            f"tl.load(index{i} + ({offset}), active, other=0)"
            if terms
            else f"tl.load(index{i})"
        )
        lines += [
            f"        raw{i} = {load}.to(tl.int64)",
            f"        normalized{i} = tl.where(raw{i} < 0, raw{i} + {shape[dim]}, raw{i})",
            f"        tl.device_assert((~active) | ((normalized{i} >= 0) "
            f'& (normalized{i} < {shape[dim]})), "index out of bounds")',
            f"        source += normalized{i} * {strides[dim]}",
            f"        active = active & (normalized{i} >= 0) & (normalized{i} < {shape[dim]})",
        ]
    lines += [
        "        value = tl.load(inp + source, active, other=0)",
        "        tl.store(out + linear, value, active)",
    ]
    plain_shape = tuple(output_shape[axis] for _, axis in plain_axes)
    plain_numel = 1
    for extent in plain_shape:
        plain_numel *= extent
    broadcast_numel = 1
    for extent in broadcast_shape:
        broadcast_numel *= extent
    span = sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    # A program owns a broadcast index and one small plain-data tile. Index
    # loads and int64 normalization are scalar, amortized across that tile.
    # Wide payloads also benefit beyond 65535 indices; narrow payloads retain
    # their existing traversal. Only bounded payload and output offsets narrow;
    # the indexed source base stays int64 when the view spans a wide address range.
    scalar_indices = (
        1 <= broadcast_numel
        and plain_numel <= 1024
        and (broadcast_numel <= 65535 or plain_numel >= 128)
        and sum((output_shape[axis] - 1) * strides[dim] for dim, axis in plain_axes)
        < 2**31
        and broadcast_numel * plain_numel < 2**31
    )
    if scalar_indices:
        out_strides = [1] * len(output_shape)
        for axis in range(len(output_shape) - 2, -1, -1):
            out_strides[axis] = out_strides[axis + 1] * output_shape[axis + 1]
        block = triton.next_power_of_2(plain_numel)
        lines = lines[:5] + [
            "def kernel(" + ", ".join(args) + "):",
            f"    for item in range(tl.program_id(0), {broadcast_numel}, PROGRAMS):",
            "        broadcast_remaining = item.to(tl.int64)",
            "        destination_base = 0",
        ]
        for dim in reversed(range(len(broadcast_shape))):
            axis = broadcast_axes[dim]
            lines += [
                f"        b{dim} = broadcast_remaining % {broadcast_shape[dim]}",
                f"        broadcast_remaining = broadcast_remaining // {broadcast_shape[dim]}",
                f"        destination_base += b{dim} * {out_strides[axis]}",
            ]
        lines += [
            "        source_base = tl.full((), 0, tl.int64)",
            "        valid_index = tl.full((), True, tl.int1)",
        ]
        for i, dim in enumerate(index_dims):
            terms = [
                f"b{j} * {stride}"
                for j, stride in enumerate(index_strides[i])
                if stride and broadcast_shape[j] > 1
            ]
            offset = " + ".join(terms) or "0"
            lines += [
                f"        raw{i} = tl.load(index{i} + ({offset})).to(tl.int64)",
                f"        normalized{i} = tl.where(raw{i} < 0, raw{i} + {shape[dim]}, raw{i})",
                f"        valid{i} = (normalized{i} >= 0) & (normalized{i} < {shape[dim]})",
                f'        tl.device_assert(valid{i}, "index out of bounds")',
                f"        source_base += tl.where(valid{i}, normalized{i}, 0) * {strides[dim]}",
                f"        valid_index = valid_index & valid{i}",
            ]
        if span < 2**31:
            lines += ["        source_base = source_base.to(tl.int32)"]
        lines += [
            "        destination_base = destination_base.to(tl.int32)",
            f"        plain = tl.arange(0, {block})",
            "        remaining = plain",
            f"        source = tl.full(({block},), 0, tl.int32)",
            "        destination = destination_base + tl.full(("
            + str(block)
            + ",), 0, tl.int32)",
        ]
        for j in reversed(range(len(plain_axes))):
            dim, axis = plain_axes[j]
            lines += [
                f"        c{j} = remaining % {output_shape[axis]}",
                f"        remaining = remaining // {output_shape[axis]}",
                f"        source += c{j} * {strides[dim]}",
                f"        destination += c{j} * {out_strides[axis]}",
            ]
        lines += [
            f"        active = (plain < {plain_numel}) & valid_index",
            "        value = tl.load(inp + source_base + source, active, other=0)",
            "        tl.store(out + destination, value, active)",
        ]

    code = "\n".join(lines) + "\n"
    key = sha256(code.encode()).hexdigest()
    path = code_cache_dir() / f"ascend_strided_index_{key}.py"
    write_atomic(path, code)
    spec = importlib.util.spec_from_file_location(f"ascend_strided_index_{key}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel, broadcast_numel if scalar_indices else None


def _index_impl(inp, indices):
    if not indices:
        raise ValueError("at least one index must be provided")
    if all(item is None for item in indices):
        raise IndexError("Index out of range")
    expanded = []
    mask_dims = set()
    for item in indices:
        if item is None:
            expanded.append(None)
        elif item.dtype in (torch.bool, torch.uint8):
            if item.ndim == 0:
                raise IndexError(
                    "zero-dimensional boolean indices are not supported by aten.index"
                )
            start = len(expanded)
            if start + item.ndim > inp.ndim or tuple(item.shape) != tuple(
                inp.shape[start : start + item.ndim]
            ):
                raise IndexError("boolean index shape does not match indexed tensor")
            expanded.extend(_boolean_coordinates(item, inp.device))
            mask_dims.update(range(start, start + item.ndim))
        elif item.dtype in (torch.int32, torch.int64):
            expanded.append(item.to(inp.device))
        else:
            raise IndexError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )
    if len(expanded) > inp.ndim:
        raise IndexError(f"too many indices for tensor of dimension {inp.ndim}")
    expanded.extend([None] * (inp.ndim - len(expanded)))
    index_dims = tuple(dim for dim, item in enumerate(expanded) if item is not None)
    tensors = [expanded[dim] for dim in index_dims]
    try:
        broadcast_shape = broadcast_shapes([item.shape for item in tensors])
    except ValueError as error:
        raise IndexError(
            "shape mismatch: indexing tensors could not be broadcast together"
        ) from error
    tensors = [item.expand(broadcast_shape) for item in tensors]
    has_output = all(broadcast_shape) and all(
        inp.size(dim) for dim in range(inp.ndim) if dim not in index_dims
    )
    if has_output:
        # Check original indices once, before their broadcast expansion.
        # Boolean coordinates already satisfy the matching shape bounds.
        for dim in index_dims:
            if dim not in mask_dims:
                _check_index_bounds(expanded[dim], inp.size(dim))
    for dim, tensor in zip(index_dims, tensors):
        expanded[dim] = tensor
    if _can_use_linearized_adjacent_index(inp, expanded):
        return _run_linearized_adjacent_index(inp, expanded)
    adjacent = index_dims == tuple(range(index_dims[0], index_dims[-1] + 1))
    plain_dims = [dim for dim in range(inp.ndim) if dim not in index_dims]
    before = (
        [dim for dim in plain_dims if index_dims and dim < index_dims[0]]
        if adjacent
        else []
    )
    after = [dim for dim in plain_dims if dim not in before]
    output_shape = (
        tuple(inp.shape[dim] for dim in before)
        + tuple(broadcast_shape)
        + tuple(inp.shape[dim] for dim in after)
    )
    broadcast_axes = tuple(range(len(before), len(before) + len(broadcast_shape)))
    plain_axes = tuple((dim, axis) for axis, dim in enumerate(before)) + tuple(
        (dim, len(before) + len(broadcast_shape) + axis)
        for axis, dim in enumerate(after)
    )
    out = torch.empty(output_shape, device=inp.device, dtype=inp.dtype)
    if out.numel():
        kernel, index_tiles = _gather_function(
            tuple(inp.shape),
            tuple(inp.stride()),
            index_dims,
            tuple(tuple(item.stride()) for item in tensors),
            tuple(broadcast_shape),
            output_shape,
            plain_axes,
            broadcast_axes,
        )
        programs = min(48, index_tiles or triton.cdiv(out.numel(), 1024))
        # Eager and capture bounds checks run before gathering. Read and write
        # masks also exclude invalid coordinates during graph replay.
        kernel[(programs,)](inp, *tensors, out, out.numel(), 1024, programs)
    return out


def index(inp, indices):
    logger.debug("GEMS_ASCEND INDEX")
    with torch_device_fn.device(inp.device):
        return _index_impl(inp, indices)
