# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import importlib
import logging
from typing import Callable, Mapping

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_put import _index_put_impl_ as _generic_index_put_impl
from flag_gems.ops.index_put import broadcast_indices, get_max_rank_shape
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _index_put_ordered_rows_kernel(
    inp,
    indices,
    values,
    n_indices: tl.constexpr,
    n_rows: tl.constexpr,
    row_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column = tl.arange(0, BLOCK)
    for update in range(n_indices):
        target_row = tl.load(indices + update)
        target_row = tl.where(target_row < 0, target_row + n_rows, target_row)
        offsets = target_row * row_width + column
        value_offsets = update * row_width + column
        value = tl.load(values + value_offsets, mask=column < row_width)
        tl.store(inp + offsets, value, mask=column < row_width)


def _can_use_last_write_rows(inp, indices, values, accumulate):
    if accumulate or len(indices) != 1:
        return False
    index = indices[0]
    if index is None or index.dtype == torch.bool or index.ndim != 1:
        return False
    if (
        not inp.is_contiguous()
        or not index.is_contiguous()
        or not values.is_contiguous()
    ):
        return False
    if inp.ndim < 2 or values.ndim != inp.ndim:
        return False
    if values.shape[0] != index.numel() or tuple(values.shape[1:]) != tuple(
        inp.shape[1:]
    ):
        return False
    return index.numel() <= 64 and inp.numel() // inp.shape[0] <= 1024


# ---------------------------------------------------------------------------
# Boolean-mask fast path: contiguous read-blend-write, no nonzero(), no atomics
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _mask_count_kernel(mask_ptr, counts_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(mask_ptr + off) != 0
    tl.store(counts_ptr + pid, tl.sum(tl.where(m, 1, 0)))


@libentry()
@triton.jit
def _mask_scan_kernel(counts_ptr, base_ptr, TILE: tl.constexpr):
    i = tl.arange(0, TILE)
    c = tl.load(counts_ptr + i)
    tl.store(base_ptr + i, tl.cumsum(c, axis=0) - c)


@libentry()
@triton.jit
def _mask_blend_kernel(
    inp_ptr,
    mask_ptr,
    values_ptr,
    base_ptr,
    IS_ACCUMULATE: tl.constexpr,
    SCALAR_VALUES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(mask_ptr + off) != 0
    cur = tl.load(inp_ptr + off)
    if SCALAR_VALUES:
        v = tl.load(values_ptr)
    else:
        one = tl.where(m, 1, 0)
        rank = tl.cumsum(one, axis=0) - one + tl.load(base_ptr + pid)
        v = tl.load(values_ptr + tl.where(m, rank, 0))
    if IS_ACCUMULATE:
        out = tl.where(m, cur + v, cur)
    else:
        out = tl.where(m, v, cur)
    tl.store(inp_ptr + off, out)


def _bool_blend_block(numel):
    """Largest power-of-two block in [64, 8192] that divides numel exactly."""
    if numel < 64:
        return 0
    block = 8192
    while block >= 64:
        if numel % block == 0:
            return block
        block //= 2
    return 0


def _can_use_bool_blend(inp, indices, values):
    if len(indices) != 1:
        return False
    mask = indices[0]
    if mask is None or mask.dtype != torch.bool:
        return False
    if tuple(mask.shape) != tuple(inp.shape):
        return False
    if not inp.is_contiguous() or not mask.is_contiguous():
        return False
    if not values.is_contiguous():
        return False
    if values.numel() != 1 and values.ndim > 1:
        return False
    return _bool_blend_block(inp.numel()) != 0


def _bool_blend(inp, mask, values, accumulate):
    numel = inp.numel()
    block = _bool_blend_block(numel)
    nb = numel // block
    tile = max(64, triton.next_power_of_2(nb + 1))
    scalar_values = values.numel() == 1

    with torch_device_fn.device(inp.device):
        if scalar_values:
            base = inp  # unused by the kernel, keep a valid pointer
        else:
            counts = torch.zeros((tile,), dtype=torch.int32, device=inp.device)
            base = torch.empty((tile,), dtype=torch.int32, device=inp.device)
            _mask_count_kernel[(nb,)](mask, counts, BLOCK=block)
            _mask_scan_kernel[(1,)](counts, base, TILE=tile)
            selected = int(base[nb].item())
            if values.numel() != selected:
                return None
        _mask_blend_kernel[(nb,)](
            inp,
            mask,
            values.view(-1) if not scalar_values else values,
            base,
            IS_ACCUMULATE=bool(accumulate),
            SCALAR_VALUES=scalar_values,
            BLOCK=block,
        )
    return inp


# ---------------------------------------------------------------------------
# Generated kernels: flat 1-D scatter (N == 1) and per-row contiguous (N > 1)
# ---------------------------------------------------------------------------


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _gen_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.newline()
    code.newline()
    return code


def _emit_decode(code, dst_prefix, src, shapes, single_is_identity):
    """Row-major decode of `src` into per-dimension coordinates."""
    q = len(shapes)
    if q == 1 and single_is_identity:
        code.writeline(f"{dst_prefix}0 = {src}")
        return
    code.writeline(f"_cur = {src}")
    for j in range(q - 1, -1, -1):
        code.writeline(f"{dst_prefix}{j} = _cur % {shapes[j]}")
        if j > 0:
            code.writeline(f"_cur = _cur // {shapes[j]}")


def _gen_flat_kernel(inp_rank, indices_len, index_rank, name, code):
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}: tl.constexpr," for i in range(inp_rank)]
        args += [f"index_shape{j}: tl.constexpr," for j in range(index_rank)]
        args += [f"input_stride{i}: tl.constexpr," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}: tl.constexpr," for j in range(index_rank)]
        args += [f"values_stride{j}: tl.constexpr," for j in range(index_rank)]
        args += [
            "M: tl.constexpr,",
            "IS_ACCUMULATE: tl.constexpr,",
            "BLOCK: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")
    with code.indent():
        code.writeline("off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
        code.writeline("m0 = off < M")
        code.writeline("safe = tl.where(m0, off, 0)")
        _emit_decode(
            code, "idx", "safe", [f"index_shape{j}" for j in range(index_rank)], True
        )
        for i in range(indices_len):
            terms = " + ".join(
                f"idx{j} * indices{i}_stride{j}" for j in range(index_rank)
            )
            code.writeline(f"ci{i} = tl.load(indices{i}_ptr + {terms})")
            code.writeline(
                f"ci{i} = tl.where(ci{i} < 0, ci{i} + input_shape{i}, ci{i})"
            )
        bounds = " & ".join(
            f"(ci{i} >= 0) & (ci{i} < input_shape{i})" for i in range(indices_len)
        )
        code.writeline(f"bok = {bounds}")
        vterms = " + ".join(f"idx{j} * values_stride{j}" for j in range(index_rank))
        code.writeline(f"vraw = tl.load(values_ptr + {vterms})")
        dterms = " + ".join(f"ci{i} * input_stride{i}" for i in range(indices_len))
        code.writeline(f"dst = tl.where(bok, {dterms}, 0)")
        # Masked stores are not honoured on this backend, so every lane must carry
        # an in-range address and an idempotent payload: lanes with off >= M
        # replicate lane 0 exactly (safe = where(m0, off, 0)), therefore the store
        # writes the very same value lane 0 writes.  `m0` is kept only to document
        # that invariant; it is intentionally not used as a store mask.
        code.writeline("_ = m0")
        code.writeline("tl.store(input_ptr + dst, vraw, mask=bok)")
    code.newline()
    code.newline()
    return code


def _gen_row_kernel(inp_rank, indices_len, index_rank, name, code):
    n_slice = inp_rank - indices_len
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}: tl.constexpr," for i in range(inp_rank)]
        args += [f"index_shape{j}: tl.constexpr," for j in range(index_rank)]
        args += [f"input_stride{i}: tl.constexpr," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}: tl.constexpr," for j in range(index_rank)]
        args += [
            f"values_stride{j}: tl.constexpr," for j in range(index_rank + n_slice)
        ]
        args += [
            "N: tl.constexpr,",
            "IS_ACCUMULATE: tl.constexpr,",
            "VCOL_LINEAR: tl.constexpr,",
            "BLOCK: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")
    with code.indent():
        code.writeline("row = tl.program_id(0)")
        code.writeline("blk = tl.program_id(1)")
        _emit_decode(
            code, "idx", "row", [f"index_shape{j}" for j in range(index_rank)], True
        )
        for i in range(indices_len):
            terms = " + ".join(
                f"idx{j} * indices{i}_stride{j}" for j in range(index_rank)
            )
            code.writeline(f"ci{i} = tl.load(indices{i}_ptr + {terms})")
            code.writeline(
                f"ci{i} = tl.where(ci{i} < 0, ci{i} + input_shape{i}, ci{i})"
            )
        bounds = " & ".join(
            f"(ci{i} >= 0) & (ci{i} < input_shape{i})" for i in range(indices_len)
        )
        code.writeline(f"ok = {bounds}")
        dterms = " + ".join(f"ci{i} * input_stride{i}" for i in range(indices_len))
        code.writeline(f"base = {dterms}")
        code.writeline("cols = blk * BLOCK + tl.arange(0, BLOCK)")
        vbase = " + ".join(f"idx{j} * values_stride{j}" for j in range(index_rank))
        code.writeline(f"vbase = {vbase}")
        code.writeline("if VCOL_LINEAR:")
        with code.indent():
            code.writeline("vcol = cols")
        code.writeline("else:")
        with code.indent():
            _emit_decode(
                code,
                "c",
                "cols",
                [f"input_shape{i}" for i in range(indices_len, inp_rank)],
                False,
            )
            vterms = " + ".join(
                f"c{j} * values_stride{index_rank + j}" for j in range(n_slice)
            )
            code.writeline(f"vcol = {vterms}")
        code.writeline("v = tl.load(values_ptr + vbase + vcol)")
        code.writeline("if ok:")
        with code.indent():
            code.writeline("if IS_ACCUMULATE:")
            with code.indent():
                code.writeline("tl.atomic_add(input_ptr + base + cols, v)")
            code.writeline("else:")
            with code.indent():
                code.writeline("tl.store(input_ptr + base + cols, v)")
    code.newline()
    code.newline()
    return code


class _GeneratedKernels:
    def __init__(self):
        self.overloads: Mapping[str, Callable] = {}

    def get(self, kind, inp_rank, indices_len, index_rank):
        key = f"{kind}_r{inp_rank}_l{indices_len}_q{index_rank}"
        fn = self.overloads.get(key)
        if fn is not None:
            return fn
        code = IndentedBuffer()
        _gen_imports(code)
        name = f"_kx_index_put_{key}"
        if kind == "flat":
            _gen_flat_kernel(inp_rank, indices_len, index_rank, name, code)
        else:
            _gen_row_kernel(inp_rank, indices_len, index_rank, name, code)
        file_path = code_cache_dir() / f"kx_index_put_impl_{key}.py"
        write_atomic(file_path, code.getvalue())
        spec = importlib.util.spec_from_file_location(
            f"_kx_ipi_mod_{key}", str(file_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, name)
        self.overloads[key] = fn
        return fn


_generated = _GeneratedKernels()

_FLAT_MIN_M = 4096


def _prod(values):
    out = 1
    for v in values:
        out *= v
    return out


def _eligible_kind(inp, indices, values, accumulate):
    """Pure metadata gate.  Must not touch tensor data or dispatch any op."""
    if not indices or any(index is None for index in indices):
        return None, None
    if len(indices) > inp.ndim:
        return None, None
    for index in indices:
        if index.dtype in (torch.bool, torch.int8):
            return None, None
        if index.device != inp.device:
            return None, None
    if values.device != inp.device:
        return None, None
    index_shape = tuple(get_max_rank_shape(list(indices)))
    if _prod(index_shape) == 0:
        return None, None
    if len(indices) == inp.ndim:
        # Discrete unmasked `tl.atomic_add` is unreliable on this backend (probe:
        # every one of 8192 unique targets off by ~1e2), so accumulate stays on the
        # generic path, which is at the atomic bandwidth wall anyway.
        if accumulate or _prod(index_shape) < _FLAT_MIN_M:
            return None, None
        return "flat", index_shape
    if not inp.is_contiguous():
        return None, None
    if _row_block(_prod(inp.shape[len(indices) :])) == 0:
        return None, None
    return "row", index_shape


def _try_flat(inp, tensor_indices, values, index_shape):
    m = _prod(index_shape)
    block = min(8192, max(64, triton.next_power_of_2(m)))
    kernel = _generated.get("flat", inp.ndim, len(tensor_indices), len(index_shape))
    args = [inp, *tensor_indices, values]
    args += list(inp.shape)
    args += list(index_shape)
    args += list(inp.stride())
    for index in tensor_indices:
        args += list(index.stride())
    args += list(values.stride())
    with torch_device_fn.device(inp.device):
        kernel[(triton.cdiv(m, block),)](*args, M=m, IS_ACCUMULATE=False, BLOCK=block)
    return inp


def _row_block(n):
    if n < 64:
        return 0
    block = 8192
    while block >= 64:
        if n % block == 0:
            return block
        block //= 2
    return 0


def _try_row(inp, tensor_indices, values, accumulate, index_shape):
    indices_len = len(tensor_indices)
    index_rank = len(index_shape)
    slice_shape = tuple(inp.shape[indices_len:])
    n = _prod(slice_shape)
    block = _row_block(n)
    m = _prod(index_shape)
    vcol_linear = tuple(values.stride()[index_rank:]) == _contiguous_strides(
        slice_shape
    )
    kernel = _generated.get("row", inp.ndim, indices_len, index_rank)
    args = [inp, *tensor_indices, values]
    args += list(inp.shape)
    args += list(index_shape)
    args += list(inp.stride())
    for index in tensor_indices:
        args += list(index.stride())
    args += list(values.stride())
    with torch_device_fn.device(inp.device):
        kernel[(m, n // block)](
            *args,
            N=n,
            IS_ACCUMULATE=bool(accumulate),
            VCOL_LINEAR=vcol_linear,
            BLOCK=block,
        )
    return inp


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    logger.debug("GEMS_KUNLUNXIN _INDEX_PUT_IMPL_")

    indices = list(indices)
    if _can_use_last_write_rows(inp, indices, values, accumulate):
        row_width = inp.numel() // inp.shape[0]
        block = min(1024, triton.next_power_of_2(row_width))
        with torch_device_fn.device(inp.device):
            _index_put_ordered_rows_kernel[(1,)](
                inp,
                indices[0],
                values,
                indices[0].numel(),
                inp.shape[0],
                row_width=row_width,
                BLOCK=block,
            )
        return inp

    if _can_use_bool_blend(inp, indices, values):
        out = _bool_blend(inp, indices[0], values, accumulate)
        if out is not None:
            return out

    kind, index_shape = _eligible_kind(inp, indices, values, accumulate)
    if kind is not None:
        norm = list(indices)
        broadcast_indices(norm, list(index_shape))
        target_shape = list(index_shape) + list(inp.shape[len(norm) :])
        val = values
        if tuple(val.shape) != tuple(target_shape):
            val = torch.broadcast_to(val, target_shape)
        if kind == "flat":
            return _try_flat(inp, norm, val, index_shape)
        return _try_row(inp, norm, val, accumulate, index_shape)

    return _generic_index_put_impl(inp, indices, values, accumulate, unsafe)
