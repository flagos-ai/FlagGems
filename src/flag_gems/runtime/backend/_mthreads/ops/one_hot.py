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

import functools
import logging
import os

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _triton_driver

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
_DEBUG_LOG = logger.isEnabledFor(logging.DEBUG)

_STRICT_VALIDATION = os.environ.get("FLAGGEMS_ONE_HOT_VALIDATE", "") not in (
    "",
    "0",
    "false",
    "False",
)

_SYNC_VALIDATE_MAX_ELEMENTS = 1024

_DENSE_MAX_OUTPUT_ELEMENTS = 67_108_864
_DENSE_MAX_CLASSES = 8192

_MINMAX_DIRECT_MAX_ELEMENTS = 10_240
_ONE_CTA_MAX_BLOCKS = 8_192
_PARTIAL_CHUNK = 4096

_INT64_MAX_ELEMENTS = 2**63 - 1


@triton.jit
def one_hot_dense_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < num_elements
    indices = tl.load(input_ptr + rows, mask=row_mask, other=0)

    cols = tl.arange(0, CLASS_BLOCK)
    for col_start in range(0, num_classes, CLASS_BLOCK):
        class_offsets = col_start + cols
        col_mask = class_offsets < num_classes
        values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
        values = values.to(tl.int64)
        out_offsets = rows[:, None] * num_classes + class_offsets[None, :]
        tl.store(
            output_ptr + out_offsets, values, mask=row_mask[:, None] & col_mask[None, :]
        )


@triton.jit
def one_hot_minmax_kernel(
    input_ptr,
    stat_ptr,
    num_elements,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    block_min = tl.min(tl.where(mask, indices, 1 << 62), axis=0)
    block_max = tl.max(tl.where(mask, indices, -(1 << 62)), axis=0)
    tl.store(stat_ptr + pid, block_min)
    tl.store(stat_ptr + num_blocks + pid, block_max)


@libentry()
@triton.jit
def one_hot_set_one_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    valid = mask & (indices >= 0) & (indices < num_classes)
    safe_indices = tl.minimum(tl.maximum(indices, 0), num_classes - 1)
    tl.store(output_ptr + offsets * num_classes + safe_indices, 1, mask=valid)


@triton.jit
def one_hot_minmax_direct_kernel(
    input_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
):
    vmin = tl.full((), 1 << 62, tl.int64)
    vmax = tl.full((), -(1 << 62), tl.int64)
    for start in range(0, num_elements, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_elements
        indices = tl.load(input_ptr + offs, mask=mask, other=0)
        vmin = tl.minimum(vmin, tl.min(tl.where(mask, indices, 1 << 62), axis=0))
        vmax = tl.maximum(vmax, tl.max(tl.where(mask, indices, -(1 << 62)), axis=0))
    tl.store(out_ptr, vmax * 2 + tl.where(vmin < 0, 1, 0))
    tl.store(out_ptr + 1, vmin)
    tl.store(out_ptr + 2, vmax)


@triton.jit
def one_hot_minmax_reduce_kernel(
    stat_ptr,
    out_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
):
    vmin = tl.full((), 1 << 62, tl.int64)
    vmax = tl.full((), -(1 << 62), tl.int64)
    for start in range(0, num_blocks, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_blocks
        lo = tl.load(stat_ptr + offs, mask=mask, other=1 << 62)
        hi = tl.load(stat_ptr + num_blocks + offs, mask=mask, other=-(1 << 62))
        vmin = tl.minimum(vmin, tl.min(lo, axis=0))
        vmax = tl.maximum(vmax, tl.max(hi, axis=0))
    tl.store(out_ptr, vmax * 2 + tl.where(vmin < 0, 1, 0))
    tl.store(out_ptr + 1, vmin)
    tl.store(out_ptr + 2, vmax)


@triton.jit
def one_hot_minmax_partial_reduce_kernel(
    stat_ptr,
    stat2_ptr,
    num_blocks,
    out_blocks,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < num_blocks
    lo = tl.load(stat_ptr + offs, mask=mask, other=1 << 62)
    hi = tl.load(stat_ptr + num_blocks + offs, mask=mask, other=-(1 << 62))
    tl.store(stat2_ptr + pid, tl.min(lo, axis=0))
    tl.store(stat2_ptr + out_blocks + pid, tl.max(hi, axis=0))


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _dense_config(num_elements: int, num_classes: int):
    class_block = min(max(_next_pow2(num_classes), 16), 256)
    hi = max(4, min(256, 4096 // class_block))
    if num_elements <= 2048:
        rows = _next_pow2(max(1, num_elements // 32))
    else:
        rows = _next_pow2(max(1, num_elements // 1024))
    rows = max(4, min(rows, hi))
    return rows, class_block, 4


def _minmax_block_size(num_elements: int):
    for block in (1024, 2048, 4096, 8192):
        size = 16 * triton.cdiv(num_elements, block)
        if size <= 256 or size >= 2048:
            return block
    return 1024


def _scatter_block_size(num_elements: int) -> int:
    if num_elements <= 1024:
        return 256
    elif num_elements <= 16384:
        return 512
    else:
        return 1024


def _fill_plan(num_elements: int, num_classes: int):
    rows, class_block, num_warps = _dense_config(num_elements, num_classes)
    grid = triton.cdiv(num_elements, rows)
    return grid, rows, class_block, num_warps


@functools.lru_cache(maxsize=4096)
def _fill_plan_cached(num_elements: int, num_classes: int):
    return _fill_plan(num_elements, num_classes)


_FILL_RUNNERS = {}
_MINMAX_RUNNERS = {}
_RUNNER_CACHE_MAX = 512

_SPEC_NUM_CLASSES = {}


def _readback_num_classes(verdict):
    packed = verdict[0].item()
    if packed & 1:
        raise RuntimeError("Class values must be non-negative.")
    if packed >= 0:
        return (packed >> 1) + 1
    return int(verdict[2].item()) + 1


def _runner_get(cache, key):
    return cache.get(key)


def _runner_put(cache, key, runner):
    if len(cache) >= _RUNNER_CACHE_MAX:
        cache.clear()
    cache[key] = runner
    return runner


def _launch_fill(
    flat_input, out, num_elements: int, num_classes: int, plan, dev, stream
):
    grid, rows, class_block, num_warps = plan
    key = (
        dev,
        flat_input.dtype,
        flat_input.data_ptr() % 16 == 0,
        out.data_ptr() % 16 == 0,
        num_elements,
        num_classes,
    )
    runner = _runner_get(_FILL_RUNNERS, key)
    if runner is not None:
        runner(flat_input, out, num_elements, num_classes, stream)
        return
    compiled = one_hot_dense_kernel[(grid,)](
        flat_input,
        out,
        num_elements,
        num_classes,
        BLOCK_SIZE=rows,
        CLASS_BLOCK=class_block,
        num_warps=num_warps,
    )
    base = compiled[(grid, 1, 1)]

    def runner(a, b, n, c, s, _b=base, _r=rows, _cb=class_block):
        _b(a, b, n, c, _r, _cb, stream=s)

    _runner_put(_FILL_RUNNERS, key, runner)


def _launch_minmax_kernels(flat_input, num_elements, target, dev, stream):
    if num_elements <= _MINMAX_DIRECT_MAX_ELEMENTS:
        key = (
            dev,
            flat_input.dtype,
            flat_input.data_ptr() % 16 == 0,
            target.data_ptr() % 16 == 0,
            num_elements,
        )
        runner = _runner_get(_MINMAX_RUNNERS, key)
        if runner is not None:
            runner(flat_input, target, num_elements, stream)
            return
        compiled = one_hot_minmax_direct_kernel[(1,)](
            flat_input,
            target,
            num_elements,
            BLOCK=8192,
            num_warps=16,
        )
        base = compiled[(1, 1, 1)]

        def runner(a, t, n, s, _b=base):
            _b(a, t, n, 8192, stream=s)

        _runner_put(_MINMAX_RUNNERS, key, runner)
        return
    BLOCK_SIZE = _minmax_block_size(num_elements)
    num_warps = max(4, min(8, BLOCK_SIZE // 1024))
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    stat = torch.empty(2 * num_blocks, device=flat_input.device, dtype=torch.int64)
    key = (
        dev,
        flat_input.dtype,
        flat_input.data_ptr() % 16 == 0,
        stat.data_ptr() % 16 == 0,
        num_elements,
        num_blocks,
        BLOCK_SIZE,
    )
    runner = _runner_get(_MINMAX_RUNNERS, key)
    if runner is None:
        compiled = one_hot_minmax_kernel[(num_blocks,)](
            flat_input,
            stat,
            num_elements,
            num_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        base = compiled[(num_blocks, 1, 1)]

        def runner(a, st, n, nb, sm, _b=base, _bs=BLOCK_SIZE):
            _b(a, st, n, nb, _bs, stream=sm)

        _runner_put(_MINMAX_RUNNERS, key, runner)
    runner(flat_input, stat, num_elements, num_blocks, stream)
    if num_blocks <= _ONE_CTA_MAX_BLOCKS:
        rkey = (
            dev,
            stat.data_ptr() % 16 == 0,
            target.data_ptr() % 16 == 0,
            num_blocks,
        )
        rrunner = _runner_get(_MINMAX_RUNNERS, rkey)
        if rrunner is not None:
            rrunner(stat, target, num_blocks, stream)
            return
        compiled = one_hot_minmax_reduce_kernel[(1,)](
            stat, target, num_blocks, BLOCK=1024
        )
        base = compiled[(1, 1, 1)]

        def rrunner(s, t, nb, sm, _b=base):
            _b(s, t, nb, 1024, stream=sm)

        _runner_put(_MINMAX_RUNNERS, rkey, rrunner)
        return
    partials = triton.cdiv(num_blocks, _PARTIAL_CHUNK)
    stat2 = torch.empty(2 * partials, device=flat_input.device, dtype=torch.int64)
    pkey = (
        dev,
        stat.data_ptr() % 16 == 0,
        stat2.data_ptr() % 16 == 0,
        num_blocks,
        partials,
    )
    prunner = _runner_get(_MINMAX_RUNNERS, pkey)
    if prunner is None:
        compiled = one_hot_minmax_partial_reduce_kernel[(partials,)](
            stat, stat2, num_blocks, partials, CHUNK=_PARTIAL_CHUNK
        )
        base = compiled[(partials, 1, 1)]

        def prunner(s, s2, nb, pb, sm, _b=base):
            _b(s, s2, nb, pb, 4096, stream=sm)

        _runner_put(_MINMAX_RUNNERS, pkey, prunner)
    prunner(stat, stat2, num_blocks, partials, stream)
    rkey = (
        dev,
        stat2.data_ptr() % 16 == 0,
        target.data_ptr() % 16 == 0,
        partials,
    )
    rrunner = _runner_get(_MINMAX_RUNNERS, rkey)
    if rrunner is None:
        compiled = one_hot_minmax_reduce_kernel[(1,)](
            stat2, target, partials, BLOCK=1024
        )
        base = compiled[(1, 1, 1)]

        def rrunner(s, t, nb, sm, _b=base):
            _b(s, t, nb, 1024, stream=sm)

        _runner_put(_MINMAX_RUNNERS, rkey, rrunner)
    rrunner(stat2, target, partials, stream)


def _one_hot_fast(tensor, flat_input, num_elements, num_classes):
    idx = tensor.device.index
    if idx is not None and torch_device_fn.current_device() == idx:
        return _one_hot_fast_impl(tensor, flat_input, num_elements, num_classes)
    with torch_device_fn.device(tensor.device):
        return _one_hot_fast_impl(tensor, flat_input, num_elements, num_classes)


def _one_hot_fast_impl(tensor, flat_input, num_elements, num_classes):
    inferred_num_classes = num_classes == -1
    device_explicit = (not inferred_num_classes) and num_classes > 0
    device = tensor.device

    active = _triton_driver.active
    cur_dev = active.get_current_device()
    cur_stream = active.get_current_stream(cur_dev)

    if inferred_num_classes:
        verdict = torch.empty(3, device=device, dtype=torch.int64)
        _launch_minmax_kernels(flat_input, num_elements, verdict, cur_dev, cur_stream)

        spec_key = (
            cur_dev,
            flat_input.dtype,
            flat_input.data_ptr() % 16 == 0,
            num_elements,
        )
        spec_nc = _SPEC_NUM_CLASSES.get(spec_key)
        spec_out = None
        if (
            spec_nc is not None
            and 1 <= spec_nc <= _DENSE_MAX_CLASSES
            and num_elements * spec_nc <= _DENSE_MAX_OUTPUT_ELEMENTS
        ):
            spec_out = torch.empty(
                (*tensor.shape, spec_nc), device=device, dtype=torch.int64
            )
            _launch_fill(
                flat_input,
                spec_out,
                num_elements,
                spec_nc,
                _fill_plan_cached(num_elements, spec_nc),
                cur_dev,
                cur_stream,
            )

        num_classes = _readback_num_classes(verdict)
        if spec_out is not None and num_classes == spec_nc:
            return spec_out
    elif not device_explicit:
        invalid = (tensor < 0) | (tensor >= num_classes)
        if invalid.any():
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
            else:
                raise RuntimeError("Class values must be smaller than num_classes.")
    elif _STRICT_VALIDATION or num_elements <= _SYNC_VALIDATE_MAX_ELEMENTS:
        host = flat_input.cpu()
        if int(host.min()) < 0:
            raise RuntimeError("Class values must be non-negative.")
        if int(host.max()) >= num_classes:
            raise RuntimeError("Class values must be smaller than num_classes.")

    if num_classes < 1:
        raise RuntimeError("num_classes should be positive")

    if num_elements * num_classes > _INT64_MAX_ELEMENTS:
        raise RuntimeError(
            "one_hot: num_elements * num_classes overflows the addressable "
            "output size"
        )

    if num_classes <= _DENSE_MAX_CLASSES and (
        num_elements * num_classes <= _DENSE_MAX_OUTPUT_ELEMENTS
    ):
        if inferred_num_classes:
            if len(_SPEC_NUM_CLASSES) >= _RUNNER_CACHE_MAX:
                _SPEC_NUM_CLASSES.clear()
            _SPEC_NUM_CLASSES[spec_key] = num_classes
        out = torch.empty(
            (*tensor.shape, num_classes), device=device, dtype=torch.int64
        )
        _launch_fill(
            flat_input,
            out,
            num_elements,
            num_classes,
            _fill_plan_cached(num_elements, num_classes),
            cur_dev,
            cur_stream,
        )
        return out

    out = torch.zeros(num_elements * num_classes, device=device, dtype=torch.int64)
    BLOCK_SIZE = _scatter_block_size(num_elements)
    one_hot_set_one_kernel[(triton.cdiv(num_elements, BLOCK_SIZE),)](
        flat_input,
        out,
        num_elements,
        num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(*tensor.shape, num_classes)


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    if _DEBUG_LOG:
        logger.debug("GEMS_MTHREADS ONE_HOT")

    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        shape = (*tensor.shape, num_classes)
        return torch.empty(shape, device=tensor.device, dtype=torch.int64)

    inferred_num_classes = num_classes == -1
    on_device = tensor.device.type != "cpu"

    if not on_device:
        if inferred_num_classes:
            num_classes = int(tensor.max().item()) + 1
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
        else:
            invalid = (tensor < 0) | (tensor >= num_classes)
            if invalid.any():
                if (tensor < 0).any():
                    raise RuntimeError("Class values must be non-negative.")
                else:
                    raise RuntimeError("Class values must be smaller than num_classes.")
        if num_classes < 1:
            raise RuntimeError("num_classes should be positive")
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    flat_input = tensor.reshape(-1)
    if not flat_input.is_contiguous():
        flat_input = flat_input.contiguous()
    return _one_hot_fast(tensor, flat_input, flat_input.numel(), num_classes)
