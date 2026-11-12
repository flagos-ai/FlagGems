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
import importlib
import inspect
import logging
import os
import time
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.device_info import get_sm_count

logger = logging.getLogger(__name__)


def _fmt_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, "
            f"stride={tuple(value.stride())}, dtype={value.dtype})"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_value(v) for v in value) + "]"
    return repr(value)


def log_args(func):
    r"""Print each argument on one line and the wall-clock execution time in ms.

    Example output:
        [index_put_] inp: Tensor(shape=(2046, 2176), stride=(2176, 1), dtype=torch.int32)
        [index_put_] indices: [Tensor(shape=(2046, 2176), stride=(2176, 1), dtype=torch.bool)]
        [index_put_] values: -1
        [index_put_] accumulate: False
        [index_put_] elapsed_ms=12.345

    grep examples:
        grep '\[index_put_\]' log.txt                     # all lines of this call
        grep '\[index_put_\] elapsed_ms' log.txt          # only the timing lines
        grep '\[index_put_\] inp:' log.txt                # only the `inp` argument
    """

    enabled = os.getenv("FLAG_GEMS_LOG_INDEX_PUT", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        prefix = f"[{func.__name__}]"
        for name, value in bound.arguments.items():
            print(f"{prefix} {name}: {_fmt_value(value)}", flush=True)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{prefix} elapsed_ms={elapsed_ms:.3f}", flush=True)

    return wrapper


# Elementwise masked-fill fast path for `inp[mask] = scalar` (single full-shape
# boolean mask, scalar value).  The generic path expands a dense bool mask via
# `nonzero()` into ~numel() integer indices and scatters them, which is
# catastrophic on the serial txda DSA (a [2046, 2176] mask -> ~4.5M scattered
# points -> ~72s single-card).  This is a plain O(numel), division-free fill.
_MFILL_BLOCK = 4096


@libentry()
@triton.jit
def _masked_fill_scalar_kernel(inp_ptr, mask_ptr, value_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num = tl.num_programs(0)
    val = tl.load(value_ptr)
    for off in range(pid * BLOCK, N, num * BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < N
        maskv = tl.load(mask_ptr + idx, mask=m)
        cur = tl.load(inp_ptr + idx, mask=m)
        out = tl.where(maskv, val.to(cur.dtype), cur)
        tl.store(inp_ptr + idx, out, mask=m)


def _masked_fill_scalar_(inp, mask, value):
    """In-place masked fill ``inp[mask] = value`` for a scalar ``value``.

    The caller gates this to a full-shape, contiguous ``mask`` and a contiguous
    ``inp`` (the GLM indexer's dense upper-triangle write), so a flat elementwise
    kernel is exact.  The scalar is read through its data pointer rather than a
    device->host ``.item()``: the txda runtime rejects 4-byte scalar copies and
    the sync alone would dominate the sub-ms kernel.
    """
    value = value if value.dtype == inp.dtype else value.to(inp.dtype)
    inp_flat = inp.view(-1)
    mask_flat = mask.view(-1)
    N = inp_flat.numel()
    if N == 0:
        return inp
    max_prog = max(1, min(get_sm_count() * 2, 128))
    grid = (min(triton.cdiv(N, _MFILL_BLOCK), max_prog),)
    _masked_fill_scalar_kernel[grid](inp_flat, mask_flat, value, N, BLOCK=_MFILL_BLOCK)
    return inp


# Tile sizes for the code-generated 2-D grid-strided kernel.  BLOCK_M tiles the
# index-entry axis (M), BLOCK_N the tail axis (N).  Both must stay powers of two
# (tl.arange requires it); the wrapper's grid function must stay in lockstep.
# Kept small enough that the 2-D int32 offset/intermediate tensors fit the txda
# shared-memory budget (~3MB) — a 128x2048 tile of int64 offsets overflows it.
BLOCK_M = 32
BLOCK_N = 1024


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
                max_num = max(max_num, index.shape[axis])
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
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")
    code.writeline("from flag_gems.utils.device_info import get_sm_count")

    code.newline()
    code.newline()
    return code


def generate_index_put_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    tail_rank = inp_rank - indices_len
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [
            f"values_stride{i}," for i in range(index_rank + inp_rank - indices_len)
        ]
        args += [
            "M,",
            "N,",
            "IN_CONTIG: tl.constexpr,",
            "VAL_TAIL: tl.constexpr,",
            "IS_ACCUMULATE: tl.constexpr,",
            "WIDE: tl.constexpr,",
            f"BLOCK_M: tl.constexpr = {BLOCK_M},",
            f"BLOCK_N: tl.constexpr = {BLOCK_N},",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        # A 2-D grid-strided kernel over (M index-entries) x (N tail elements).
        # The two program ids directly give the index-entry block and the tail
        # block, so no flat-id -> (row, col) division is needed at all.  That
        # division (offs // N, offs % N) is *catastrophic* on the serial txda DSA
        # when N is not a power of two (e.g. 2176): ~360ms for a 17MB write vs
        # 0.45ms for the division-free 2-D form.  We only divide when the index
        # rank or the tail rank exceed 1 (i.e. the *inner* dims), and even then it
        # is usually by a power-of-two dim (head_dim, page_size, ...).
        code.writeline("pid_m = tl.program_id(0)")
        code.writeline("pid_n = tl.program_id(1)")
        code.writeline("num_m = tl.num_programs(0)")
        code.writeline("num_n = tl.num_programs(1)")
        code.writeline("for m_start in range(pid_m * BLOCK_M, M, num_m * BLOCK_M):")
        with code.indent():
            code.writeline("offset0 = m_start + tl.arange(0, BLOCK_M)")
            # WIDE: flat offsets can exceed int32 (numel > 2^31); compute every
            # offset product in int64 so tl.store cannot wrap to a wild address.
            code.writeline("if WIDE:")
            with code.indent():
                code.writeline("offset0 = offset0.to(tl.int64)")
            code.writeline("mask0 = offset0 < M")
            # Decompose the index-entry id into per-index-tensor coordinates.  The
            # outermost coord is the remainder, so we only divide for inner dims.
            code.writeline("cur_m = offset0")
            for i in range(index_rank - 1, 0, -1):
                code.writeline(f"indices_idx{i} = cur_m % indices0_shape{i}")
                code.writeline(f"cur_m = cur_m // indices0_shape{i}")
            code.writeline("indices_idx0 = cur_m")
            code.newline()
            code.writeline("for n_start in range(pid_n * BLOCK_N, N, num_n * BLOCK_N):")
            with code.indent():
                code.writeline("offset1 = n_start + tl.arange(0, BLOCK_N)")
                code.writeline("if WIDE:")
                with code.indent():
                    code.writeline("offset1 = offset1.to(tl.int64)")
                code.writeline("mask1 = offset1 < N")
                # Compute the tail contribution to input_offset / values_offset.
                # Decomposing the flat tail id (offset1) into the trailing input
                # dims needs `%` / `//`, which is catastrophically slow on txda
                # (~300x: a single 4-D write of a contiguous (2, 128, 128) tail
                # costs ~75ms with the division vs ~0.3ms without).  When the
                # tail strides are the natural row-major strides the flat offset1
                # *is* the tail offset, and when they are all zero (a broadcast
                # scalar value) the tail contribution is simply 0 -- in both cases
                # the division can be skipped entirely.
                if tail_rank > 1:
                    code.writeline("if IN_CONTIG and VAL_TAIL != 2:")
                    with code.indent():
                        code.writeline("in_tail = offset1[None, :]")
                        code.writeline("if VAL_TAIL == 0:")
                        with code.indent():
                            code.writeline("val_tail = offset1[None, :]")
                        code.writeline("else:")
                        with code.indent():
                            # VAL_TAIL == 1: broadcast (zero-stride) values tail.
                            code.writeline("val_tail = offset1[None, :] * 0")
                    code.writeline("else:")
                    with code.indent():
                        code.writeline("cur_n = offset1")
                        for i in range(inp_rank - 1, indices_len, -1):
                            code.writeline(f"input_idx{i} = cur_n % input_shape{i}")
                            code.writeline(f"cur_n = cur_n // input_shape{i}")
                        code.writeline(f"input_idx{indices_len} = cur_n")
                        code.writeline("if IN_CONTIG:")
                        with code.indent():
                            code.writeline("in_tail = offset1[None, :]")
                        code.writeline("else:")
                        with code.indent():
                            in_tail_terms = [
                                f"input_idx{i}[None, :] * input_stride{i}"
                                for i in range(indices_len, inp_rank)
                            ]
                            code.writeline(f"in_tail = {' + '.join(in_tail_terms)}")
                        code.writeline("if VAL_TAIL == 0:")
                        with code.indent():
                            code.writeline("val_tail = offset1[None, :]")
                        code.writeline("elif VAL_TAIL == 1:")
                        with code.indent():
                            code.writeline("val_tail = offset1[None, :] * 0")
                        code.writeline("else:")
                        with code.indent():
                            val_tail_terms = [
                                f"input_idx{indices_len + i}[None, :] * values_stride{index_rank + i}"
                                for i in range(tail_rank)
                            ]
                            code.writeline(f"val_tail = {' + '.join(val_tail_terms)}")
                elif tail_rank == 1:
                    code.writeline(f"input_idx{indices_len} = offset1")
                    code.writeline(
                        f"in_tail = input_idx{indices_len}[None, :] * input_stride{indices_len}"
                    )
                    code.writeline(
                        f"val_tail = input_idx{indices_len}[None, :] * values_stride{index_rank}"
                    )
                else:
                    # tail_rank == 0 (pure scalar scatter, N == 1): force the 2-D
                    # [BLOCK_M, BLOCK_N] shape so the store broadcasts correctly.
                    code.writeline("in_tail = offset1[None, :] * 0")
                    code.writeline("val_tail = offset1[None, :] * 0")
                code.newline()
                for i in range(indices_len):
                    comp = [
                        f"indices_idx{j} * indices{i}_stride{j}"
                        for j in range(index_rank)
                    ]
                    # Cast to int32: index coords always fit (dim < 2^31), and it
                    # keeps the offset arithmetic native on txda (int64 is emulated)
                    # and halves the 2-D intermediate shared-memory footprint.
                    # WIDE (input numel > 2^31): keep int64 -- the *product*
                    # cur_index * input_stride can exceed int32 even when the
                    # coordinate itself fits (e.g. 47 * 82968576 ~ 3.9e9).
                    load_expr = f"tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0)"
                    code.writeline("if WIDE:")
                    with code.indent():
                        code.writeline(f"cur_index{i} = {load_expr}.to(tl.int64)")
                    code.writeline("else:")
                    with code.indent():
                        code.writeline(f"cur_index{i} = {load_expr}.to(tl.int32)")
                code.newline()
                index_mask = [
                    f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
                    for i in range(indices_len)
                ]
                code.writeline(f"index_mask = {' & '.join(index_mask)}")
                code.writeline("shape_mask = mask0[:, None] & mask1[None, :]")
                code.writeline("mask = shape_mask & index_mask[:, None]")
                code.newline()
                in_lead = " + ".join(
                    f"cur_index{i}[:, None] * input_stride{i}"
                    for i in range(indices_len)
                )
                val_lead = " + ".join(
                    f"indices_idx{i}[:, None] * values_stride{i}"
                    for i in range(index_rank)
                )
                code.writeline(f"input_offset = {in_lead} + in_tail")
                code.writeline(f"values_offset = {val_lead} + val_tail")
                code.newline()
                # Load values.  VAL_TAIL == 1 (broadcast tail) means the value at
                # (m, n) depends only on the index-entry coord m, so a [BLOCK_M]
                # load broadcast to [BLOCK_M, BLOCK_N] avoids the ~60x-slow 2-D
                # gather a 0-stride tensor triggers on txda (12.9ms -> 0.2ms).
                # Otherwise load the full 2-D block with the shape-only mask: a
                # data-dependent (index-bounds) mask on this load is what made
                # txda ~300x slow.
                code.writeline("if VAL_TAIL == 1:")
                with code.indent():
                    val_lead_1d = " + ".join(
                        f"indices_idx{i} * values_stride{i}" for i in range(index_rank)
                    )
                    code.writeline(
                        f"cur_value = tl.load(values_ptr + {val_lead_1d}, mask=mask0, other=0)[:, None]"
                    )
                code.writeline("else:")
                with code.indent():
                    code.writeline(
                        "cur_value = tl.load(values_ptr + values_offset, mask=shape_mask)"
                    )
                code.writeline("if IS_ACCUMULATE:")
                with code.indent():
                    # txda's tl.atomic_add rejects a `mask=` argument (lowering
                    # error: "'tensor.empty' op incorrect number of dynamic
                    # sizes").  Mask instead by clamping the destination to a
                    # safe position (0) and the value to a real zero of the same
                    # dtype, so masked-out lanes add exactly 0 to a valid
                    # location.  tl.zeros(...) rather than `cur_value * 0`
                    # keeps a masked-out NaN from polluting position 0
                    # (NaN * 0 == NaN).
                    code.writeline("safe_offset = tl.where(mask, input_offset, 0)")
                    code.writeline(
                        "safe_value = tl.where(mask, cur_value, "
                        "tl.zeros((BLOCK_M, BLOCK_N), cur_value.dtype))"
                    )
                    code.writeline(
                        'tl.atomic_add(input_ptr + safe_offset, safe_value, sem="relaxed")'
                    )
                code.writeline("else:")
                with code.indent():
                    code.writeline(
                        "tl.store(input_ptr + input_offset, cur_value, mask=mask)"
                    )

    code.newline()
    code.newline()
    return code


def generate_index_put_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    tail_rank = inp_rank - indices_len
    code.writeline(f"def {wrapper_name}(input, indices, values, accumulate):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("values_shape = values.shape")
        code.writeline("values_stride = values.stride()")
        # Classify the tail dims (dims [indices_len:] of input, dims [index_rank:]
        # of values) so the kernel can skip the slow integer div/mod when they are
        # row-major contiguous (IN_CONTIG / VAL_TAIL == 0) or a broadcast scalar
        # (VAL_TAIL == 1).  Otherwise (VAL_TAIL == 2) the div/mod decomposition is
        # still used and stays correct, just slower.
        code.writeline("_acc = 1")
        code.writeline("IN_CONTIG = True")
        code.writeline(f"for _k in range({inp_rank} - 1, {indices_len} - 1, -1):")
        with code.indent():
            code.writeline("if input_stride[_k] != _acc:")
            with code.indent():
                code.writeline("IN_CONTIG = False")
            code.writeline("_acc *= input_shape[_k]")
        code.writeline("_acc = 1")
        code.writeline("_val_contig = True")
        code.writeline("_val_zero = True")
        code.writeline(
            f"for _k in range({index_rank + tail_rank} - 1, {index_rank} - 1, -1):"
        )
        with code.indent():
            code.writeline("if values_stride[_k] != _acc:")
            with code.indent():
                code.writeline("_val_contig = False")
            code.writeline("if values_stride[_k] != 0:")
            with code.indent():
                code.writeline("_val_zero = False")
            code.writeline("_acc *= values_shape[_k]")
        code.writeline("VAL_TAIL = 0 if _val_contig else (1 if _val_zero else 2)")
        # WIDE: if the largest possible flat element offset of the input (or the
        # values) can exceed int32, run the kernel with int64 offset arithmetic.
        # A pure int32 pipeline silently wraps for tensors with numel > 2^31
        # (e.g. a (48, 844, 6, 128, 128) float32 view: 47 * 82968576 ~ 3.9e9)
        # and the store lands at a wild address, corrupting unrelated memory.
        code.writeline("_int32_max = 2147483647")
        code.writeline("_max_in_off = 0")
        code.writeline(f"for _i in range({inp_rank}):")
        with code.indent():
            code.writeline("_max_in_off += (input_shape[_i] - 1) * input_stride[_i]")
        code.writeline("_max_val_off = 0")
        code.writeline(f"for _i in range({index_rank + inp_rank - indices_len}):")
        with code.indent():
            code.writeline("_max_val_off += (values_shape[_i] - 1) * values_stride[_i]")
        code.writeline(
            "WIDE = (_max_in_off > _int32_max) or (_max_val_off > _int32_max)"
        )
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.writeline("if M == 0 or N == 0:")
        with code.indent():
            code.writeline("return input")
        # Bound the program count per axis.  txda serialises program launches, so
        # thousands of tiny programs is catastrophic (~0.2ms each); a few dozen
        # grid-striding programs saturate both txda (16 cores) and CUDA SMs.
        # Cap the 2-D product at max_prog so a huge M *and* huge N cannot fan out.
        code.writeline("max_prog = max(1, min(get_sm_count() * 2, 128))")
        code.writeline(f"grid_m = min(triton.cdiv(M, {BLOCK_M}), max_prog)")
        code.writeline(
            f"grid_n = min(triton.cdiv(N, {BLOCK_N}), max(1, max_prog // grid_m))"
        )
        code.writeline("grid = lambda meta: (grid_m, grid_n)")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["values,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"values_stride[{i}],"
                for i in range(index_rank + inp_rank - indices_len)
            ]
            args += [
                "M,",
                "N,",
                "IN_CONTIG,",
                "VAL_TAIL,",
                "accumulate==True,",
                "WIDE,",
            ]
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
    generate_index_put_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_index_put_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexPutFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, values, accumulate = args
        full_args = (inp, tensor_indices, values)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
                "_index_put_wrapper",
                "_index_put_jit_function",
                code,
            )
            file_name = f"index_put_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_put_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices, _ = args[0], args[1], args[2]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        if indices_len == 0:
            index_rank = 0
        else:
            index_rank = tensor_indices[0].ndim
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_put_func = IndexPutFunction()


def index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS_TSINGMICRO INDEX_PUT")

    out = inp.clone()
    return index_put_(out, indices, values, accumulate)


@log_args
def index_put_(inp, indices, values, accumulate=False):
    logger.debug("GEMS_TSINGMICRO INDEX_PUT_")

    indices = list(indices)

    if not indices:
        raise ValueError("At least one index tensor is required")

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]

    # Fast path: `inp[mask] = scalar` (a single full-shape bool mask + scalar
    # value) is a masked fill, not a scatter.  Expanding a dense bool mask via
    # `nonzero()` into ~numel() integer indices and scattering them is
    # catastrophically slow on txda (a [2046, 2176] mask -> ~4.5M scattered
    # points -> ~72s).  Route it to an elementwise masked-fill kernel (~0.6ms)
    # before the nonzero expansion.  Gated to full-shape + contiguous so any
    # broadcast/strided mask still takes the general (correct) path below.
    if (
        not accumulate
        and len(indices) == 1
        and isinstance(indices[0], torch.Tensor)
        and indices[0].dtype == torch.bool
        and indices[0].shape == inp.shape
        and indices[0].is_contiguous()
        and inp.is_contiguous()
        and isinstance(values, torch.Tensor)
        and values.numel() == 1
    ):
        return _masked_fill_scalar_(inp, indices[0], values)

    # step 1: index preprocessing
    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.int8):
            # Expand bool masks into explicit integer indices
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif torch.is_tensor(idx):
            processed_indices.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    indices = processed_indices
    # Pad missing None indices to match input dimension
    if len(indices) < inp.ndim:
        indices.extend([None] * (inp.ndim - len(indices)))

    if len(indices) > inp.ndim:
        raise IndexError("too many indices for tensor of dimension {}".format(inp.ndim))

    # Step 2: Broadcast tensor indices
    tensor_pos = [i for i, x in enumerate(indices) if x is not None]
    if not tensor_pos:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [indices[i] for i in tensor_pos]
    if len(tensor_indices) > 1:
        broadcasted = torch.broadcast_tensors(*tensor_indices)
        for i, pos in enumerate(tensor_pos):
            indices[pos] = broadcasted[i]

    # Step 3: Transpose
    is_contiguous = (tensor_pos[-1] - tensor_pos[0] + 1) == len(tensor_pos)
    starts_with_none = indices[0] is None
    need_transpose = not is_contiguous or starts_with_none

    if need_transpose:
        perm_order = tensor_pos + [i for i, x in enumerate(indices) if x is None]
        inp_view = inp.permute(perm_order)
        final_indices = [indices[i] for i in tensor_pos] + [None] * (
            len(indices) - len(tensor_pos)
        )
    else:
        inp_view = inp
        final_indices = indices

    # Step 4: Handle Values shape and broadcasting
    tensors = [x for x in final_indices if x is not None]
    broadcast_shape = list(tensors[0].shape)
    slice_shape = [inp_view.shape[i] for i, x in enumerate(final_indices) if x is None]

    target_shape = broadcast_shape + slice_shape
    values = values.to(inp.device)
    if need_transpose and is_contiguous:
        num_before = tensor_pos[0]

        # 1. Broadcast to PyTorch natural shape
        before_dims = slice_shape[:num_before]
        after_dims = slice_shape[num_before:]
        natural_shape = before_dims + broadcast_shape + after_dims
        values = values.broadcast_to(natural_shape)

        # 2. Permute to Kernel expectation
        B, T = len(before_dims), len(broadcast_shape)
        val_perm = (
            list(range(B, B + T)) + list(range(0, B)) + list(range(B + T, values.ndim))
        )
        values = values.permute(val_perm)
    else:
        # direct broadcast
        values = values.broadcast_to(target_shape)

    _index_put_func(inp_view, tensors, values, accumulate)

    return inp


@log_args
def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    logger.debug("GEMS_TSINGMICRO _INDEX_PUT_IMPL_")

    # The `unsafe` parameter is a hint to PyTorch for bounds checking.
    # Our implementation always performs bounds checking, so we ignore this parameter.
    # This is consistent with how PyTorch handles it internally.

    indices = list(indices)

    if len(indices) == 1 and indices[0].dtype == torch.bool:
        mask = indices[0]

        if mask.device != inp.device:
            mask = mask.to(inp.device)

        indices = list(torch.where(mask))

        K = indices[0].numel()
        target_shape = (K,) + inp.shape[len(indices) :]

        if values.numel() == 1:
            values = torch.full(
                target_shape, values.item(), dtype=inp.dtype, device=inp.device
            )
        elif values.numel() == K:
            values = values.reshape((K,)).expand(target_shape)

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]

    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)
    target_shape += inp.shape[len(indices) :]
    # Filter out None values for kernel call (only tensor indices)
    # Must be done AFTER broadcast_indices, as broadcast may create new tensors
    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        raise ValueError("At least one non-None index tensor is required")

    if values.device != inp.device:
        values = values.to(inp.device)
    values = torch.broadcast_to(values, target_shape)

    _index_put_func(inp, tensor_indices, values, accumulate)
    return inp
