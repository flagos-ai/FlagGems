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

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def get_max_rank_shape(indices: List[torch.Tensor]) -> List[int]:
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
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("import triton.experimental.tle.language as tle")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")
    code.newline()
    code.newline()
    return code


def generate_index_put_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")

    is_1d = inp_rank == indices_len

    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        if is_1d:
            pass
        else:
            args += ["copy_src_ptr,", "COPY_N,"]
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
            "IS_ACCUMULATE: tl.constexpr,",
            "BLOCK_SIZE0: tl.constexpr,",
        ]
        if is_1d:
            args += ["UNROLL: tl.constexpr = 1,"]
        else:
            args += [
                "BLOCK_SIZE1: tl.constexpr = 1024,",
                "IS_INP_CONTIGUOUS: tl.constexpr = True,",
                "V_MODE: tl.constexpr = 1,",
                "V_CHUNK: tl.constexpr = 1,",
                "USE_ASYNC: tl.constexpr = True,",
                "SWAP_GRID: tl.constexpr = False,",
                "LOOP: tl.constexpr = 1,",
                "EXACT: tl.constexpr = False,",
                "TINY: tl.constexpr = False,",
                "BLOCK_M: tl.constexpr = 1,",
                "BLOCK_N: tl.constexpr = 1,",
                "FUSE_COPY: tl.constexpr = False,",
                "COPY_BLOCK: tl.constexpr = 1,",
            ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        if is_1d:
            code.writeline("pid0 = ext.program_id(axis=0)")

        if is_1d:
            code.writeline(
                "base_offs = pid0 * BLOCK_SIZE0 * UNROLL + tl.arange(0, BLOCK_SIZE0)"
            )
            code.writeline("if UNROLL == 4:")
            with code.indent():
                for u in range(4):
                    code.writeline(f"offs_{u} = base_offs + {u} * BLOCK_SIZE0")
                    code.writeline(f"mask_{u} = offs_{u} < M")
                code.newline()

                if index_rank == 1:
                    for u in range(4):
                        code.writeline(f"idx_pos_{u} = offs_{u}")
                else:
                    for u in range(4):
                        code.writeline(f"cur_idx_{u} = offs_{u}")
                        for j in range(index_rank - 1, -1, -1):
                            code.writeline(
                                f"indices_idx{j}_{u} = cur_idx_{u} % indices0_shape{j}"
                            )
                            code.writeline(
                                f"cur_idx_{u} = cur_idx_{u} // indices0_shape{j}"
                            )
                code.newline()

                for i in range(indices_len):
                    for u in range(4):
                        if index_rank == 1:
                            addr_comp = f"offs_{u} * indices{i}_stride0"
                        else:
                            addr_comp = " + ".join(
                                [
                                    f"indices_idx{j}_{u} * indices{i}_stride{j}"
                                    for j in range(index_rank)
                                ]
                            )
                        code.writeline(
                            f"cur_index{i}_{u} = tle.load(indices{i}_ptr + {addr_comp}, "
                            f"mask=mask_{u}, other=0, is_async=True)"
                        )
                code.newline()

                for u in range(4):
                    conds = [
                        f"(cur_index{i}_{u} >= 0) & (cur_index{i}_{u} < input_shape{i})"
                        for i in range(indices_len)
                    ]
                    code.writeline(f"valid_{u} = mask_{u} & {' & '.join(conds)}")
                code.newline()

                for u in range(4):
                    comp = [
                        f"cur_index{i}_{u} * input_stride{i}"
                        for i in range(indices_len)
                    ]
                    code.writeline(f"input_offset_{u} = {' + '.join(comp)}")
                code.newline()

                for u in range(4):
                    if index_rank == 1:
                        val_comp = f"offs_{u} * values_stride0"
                    else:
                        val_comp = " + ".join(
                            [
                                f"indices_idx{j}_{u} * values_stride{j}"
                                for j in range(index_rank)
                            ]
                        )
                    code.writeline(
                        f"val_{u} = tle.load(values_ptr + {val_comp}, mask=valid_{u}, is_async=True)"
                    )
                code.newline()

                code.writeline("if IS_ACCUMULATE:")
                with code.indent():
                    for u in range(4):
                        code.writeline(
                            f"tl.atomic_add(input_ptr + input_offset_{u}, val_{u}, mask=valid_{u}, sem='relaxed')"
                        )
                code.writeline("else:")
                with code.indent():
                    for u in range(4):
                        code.writeline(
                            f"tl.store(input_ptr + input_offset_{u}, val_{u}, mask=valid_{u})"
                        )

            code.writeline("elif UNROLL == 2:")
            with code.indent():
                for u in range(2):
                    code.writeline(f"offs_{u} = base_offs + {u} * BLOCK_SIZE0")
                    code.writeline(f"mask_{u} = offs_{u} < M")
                code.newline()

                if index_rank == 1:
                    for u in range(2):
                        code.writeline(f"idx_pos_{u} = offs_{u}")
                else:
                    for u in range(2):
                        code.writeline(f"cur_idx_{u} = offs_{u}")
                        for j in range(index_rank - 1, -1, -1):
                            code.writeline(
                                f"indices_idx{j}_{u} = cur_idx_{u} % indices0_shape{j}"
                            )
                            code.writeline(
                                f"cur_idx_{u} = cur_idx_{u} // indices0_shape{j}"
                            )
                code.newline()

                for i in range(indices_len):
                    for u in range(2):
                        if index_rank == 1:
                            addr_comp = f"offs_{u} * indices{i}_stride0"
                        else:
                            addr_comp = " + ".join(
                                [
                                    f"indices_idx{j}_{u} * indices{i}_stride{j}"
                                    for j in range(index_rank)
                                ]
                            )
                        code.writeline(
                            f"cur_index{i}_{u} = tle.load(indices{i}_ptr + {addr_comp}, "
                            f"mask=mask_{u}, other=0, is_async=True)"
                        )
                code.newline()

                for u in range(2):
                    conds = [
                        f"(cur_index{i}_{u} >= 0) & (cur_index{i}_{u} < input_shape{i})"
                        for i in range(indices_len)
                    ]
                    code.writeline(f"valid_{u} = mask_{u} & {' & '.join(conds)}")
                code.newline()

                for u in range(2):
                    comp = [
                        f"cur_index{i}_{u} * input_stride{i}"
                        for i in range(indices_len)
                    ]
                    code.writeline(f"input_offset_{u} = {' + '.join(comp)}")
                code.newline()

                for u in range(2):
                    if index_rank == 1:
                        val_comp = f"offs_{u} * values_stride0"
                    else:
                        val_comp = " + ".join(
                            [
                                f"indices_idx{j}_{u} * values_stride{j}"
                                for j in range(index_rank)
                            ]
                        )
                    code.writeline(
                        f"val_{u} = tle.load(values_ptr + {val_comp}, mask=valid_{u}, is_async=True)"
                    )
                code.newline()

                code.writeline("if IS_ACCUMULATE:")
                with code.indent():
                    for u in range(2):
                        code.writeline(
                            f"tl.atomic_add(input_ptr + input_offset_{u}, val_{u}, mask=valid_{u}, sem='relaxed')"
                        )
                code.writeline("else:")
                with code.indent():
                    for u in range(2):
                        code.writeline(
                            f"tl.store(input_ptr + input_offset_{u}, val_{u}, mask=valid_{u})"
                        )

            code.writeline("else:")
            with code.indent():
                code.writeline("offs = base_offs")
                code.writeline("mask = offs < M")
                if index_rank == 1:
                    code.writeline("idx_pos = offs")
                else:
                    code.writeline("cur_idx = offs")
                    for j in range(index_rank - 1, -1, -1):
                        code.writeline(f"indices_idx{j} = cur_idx % indices0_shape{j}")
                        code.writeline(f"cur_idx = cur_idx // indices0_shape{j}")
                code.newline()

                for i in range(indices_len):
                    if index_rank == 1:
                        addr_comp = f"offs * indices{i}_stride0"
                    else:
                        addr_comp = " + ".join(
                            [
                                f"indices_idx{j} * indices{i}_stride{j}"
                                for j in range(index_rank)
                            ]
                        )
                    code.writeline(
                        f"cur_index{i} = tle.load(indices{i}_ptr + {addr_comp}, mask=mask, other=0, is_async=True)"
                    )
                conds = [
                    f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
                    for i in range(indices_len)
                ]
                code.writeline(f"valid = mask & {' & '.join(conds)}")
                comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
                code.writeline(f"input_offset = {' + '.join(comp)}")

                if index_rank == 1:
                    val_comp = "offs * values_stride0"
                else:
                    val_comp = " + ".join(
                        [
                            f"indices_idx{j} * values_stride{j}"
                            for j in range(index_rank)
                        ]
                    )
                code.writeline(
                    f"val = tle.load(values_ptr + {val_comp}, mask=valid, is_async=True)"
                )

                code.writeline("if IS_ACCUMULATE:")
                with code.indent():
                    code.writeline(
                        "tl.atomic_add(input_ptr + input_offset, val, mask=valid, sem='relaxed')"
                    )
                code.writeline("else:")
                with code.indent():
                    code.writeline(
                        "tl.store(input_ptr + input_offset, val, mask=valid)"
                    )
        else:
            trailing_dims = inp_rank - indices_len

            def emit_body(mask_expr):
                mk = "" if mask_expr is None else f", mask={mask_expr}"
                code.writeline("if IS_INP_CONTIGUOUS:")
                with code.indent():
                    code.writeline("input_offset = input_slice_base + offs")
                code.writeline("else:")
                with code.indent():
                    code.writeline("cur_n = offs")
                    for i in range(inp_rank - 1, indices_len - 1, -1):
                        code.writeline(f"input_idx{i} = cur_n % input_shape{i}")
                        code.writeline(f"cur_n = cur_n // input_shape{i}")
                    comp_c = " + ".join(
                        f"input_idx{i} * input_stride{i}"
                        for i in range(indices_len, inp_rank)
                    )
                    code.writeline(f"input_offset = input_slice_base + {comp_c}")
                code.newline()
                code.writeline("if V_MODE == 0:")
                with code.indent():
                    code.writeline("cur_value = tl.load(values_ptr)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("if V_MODE == 1:")
                    with code.indent():
                        code.writeline("values_offset = values_slice_base + offs")
                    code.writeline("elif V_MODE == 2:")
                    with code.indent():
                        code.writeline(
                            "values_offset = values_slice_base + offs % V_CHUNK"
                        )
                    code.writeline("else:")
                    with code.indent():
                        code.writeline("cur_nv = offs")
                        for i in range(inp_rank - 1, indices_len - 1, -1):
                            code.writeline(f"val_idx{i} = cur_nv % input_shape{i}")
                            code.writeline(f"cur_nv = cur_nv // input_shape{i}")
                        comp_v = " + ".join(
                            f"val_idx{indices_len + i} * values_stride{index_rank + i}"
                            for i in range(trailing_dims)
                        )
                        code.writeline(f"values_offset = values_slice_base + {comp_v}")
                    code.newline()
                    code.writeline("if USE_ASYNC:")
                    with code.indent():
                        code.writeline(
                            f"cur_value = tle.load(values_ptr + values_offset{mk}, is_async=True)"
                        )
                    code.writeline("else:")
                    with code.indent():
                        code.writeline(
                            f"cur_value = tl.load(values_ptr + values_offset{mk})"
                        )
                code.newline()
                code.writeline("if IS_ACCUMULATE:")
                with code.indent():
                    code.writeline(
                        f"tl.atomic_add(input_ptr + input_offset, cur_value{mk}, sem='relaxed')"
                    )
                code.writeline("else:")
                with code.indent():
                    code.writeline(
                        f"tl.store(input_ptr + input_offset, cur_value{mk}, cache_modifier='.cg')"
                    )

            code.writeline("if FUSE_COPY:")
            with code.indent():
                code.writeline("coffs = tl.arange(0, COPY_BLOCK)")
                code.writeline("cmask = coffs < COPY_N")
                code.writeline(
                    "tl.store(input_ptr + coffs, tl.load(copy_src_ptr + coffs, mask=cmask), mask=cmask)"
                )
                code.writeline("tl.debug_barrier()")
            code.newline()

            code.writeline("if SWAP_GRID:")
            with code.indent():
                code.writeline("pid0 = ext.program_id(axis=1)")
                code.writeline("pid1 = ext.program_id(axis=0)")
            code.writeline("else:")
            with code.indent():
                code.writeline("pid0 = ext.program_id(axis=0)")
                code.writeline("pid1 = ext.program_id(axis=1)")

            code.writeline("if TINY:")
            with code.indent():
                code.writeline("midx = tl.arange(0, BLOCK_M)")
                code.writeline("mmask = midx < M")
                code.writeline("cur_m = midx")
                for i in range(index_rank - 1, -1, -1):
                    code.writeline(f"indices_idx{i} = cur_m % indices0_shape{i}")
                    code.writeline(f"cur_m = cur_m // indices0_shape{i}")
                code.newline()
                for i in range(indices_len):
                    comp = " + ".join(
                        f"indices_idx{j} * indices{i}_stride{j}"
                        for j in range(index_rank)
                    )
                    code.writeline(
                        f"cur_index{i} = tl.load(indices{i}_ptr + {comp}, mask=mmask, other=0)"
                    )
                bounds = " & ".join(
                    f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
                    for i in range(indices_len)
                )
                code.writeline(f"valid_slice = mmask & {bounds}")
                code.writeline(
                    "input_row_base = "
                    + " + ".join(
                        f"cur_index{i} * input_stride{i}" for i in range(indices_len)
                    )
                )
                code.writeline(
                    "values_row_base = "
                    + " + ".join(
                        f"indices_idx{j} * values_stride{j}" for j in range(index_rank)
                    )
                )
                code.newline()
                code.writeline("noffs = tl.arange(0, BLOCK_N)")
                code.writeline("if IS_INP_CONTIGUOUS:")
                with code.indent():
                    code.writeline("input_col_off = noffs")
                code.writeline("else:")
                with code.indent():
                    code.writeline("cur_n = noffs")
                    for i in range(inp_rank - 1, indices_len - 1, -1):
                        code.writeline(f"input_idx{i} = cur_n % input_shape{i}")
                        code.writeline(f"cur_n = cur_n // input_shape{i}")
                    comp_c = " + ".join(
                        f"input_idx{i} * input_stride{i}"
                        for i in range(indices_len, inp_rank)
                    )
                    code.writeline(f"input_col_off = {comp_c}")
                code.newline()
                code.writeline("mask2 = valid_slice[:, None] & (noffs < N)[None, :]")
                code.writeline(
                    "input_offset = input_row_base[:, None] + input_col_off[None, :]"
                )
                code.newline()
                code.writeline("if V_MODE == 0:")
                with code.indent():
                    code.writeline("cur_value = tl.load(values_ptr)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("if V_MODE == 1:")
                    with code.indent():
                        code.writeline("values_col_off = noffs")
                    code.writeline("elif V_MODE == 2:")
                    with code.indent():
                        code.writeline("values_col_off = noffs % V_CHUNK")
                    code.writeline("else:")
                    with code.indent():
                        code.writeline("cur_nv = noffs")
                        for i in range(inp_rank - 1, indices_len - 1, -1):
                            code.writeline(f"val_idx{i} = cur_nv % input_shape{i}")
                            code.writeline(f"cur_nv = cur_nv // input_shape{i}")
                        comp_v = " + ".join(
                            f"val_idx{indices_len + i} * values_stride{index_rank + i}"
                            for i in range(trailing_dims)
                        )
                        code.writeline(f"values_col_off = {comp_v}")
                    code.writeline(
                        "cur_value = tl.load(values_ptr + values_row_base[:, None] "
                        "+ values_col_off[None, :], mask=mask2)"
                    )
                code.newline()
                code.writeline("if IS_ACCUMULATE:")
                with code.indent():
                    code.writeline(
                        "tl.atomic_add(input_ptr + input_offset, cur_value, mask=mask2, sem='relaxed')"
                    )
                code.writeline("else:")
                with code.indent():
                    code.writeline(
                        "tl.store(input_ptr + input_offset, cur_value, mask=mask2)"
                    )

            code.writeline("else:")
            with code.indent():
                code.writeline("cur_m = pid0")
                for i in range(index_rank - 1, -1, -1):
                    code.writeline(f"indices_idx{i} = cur_m % indices0_shape{i}")
                    code.writeline(f"cur_m = cur_m // indices0_shape{i}")
                code.newline()
                for i in range(indices_len):
                    comp = " + ".join(
                        f"indices_idx{j} * indices{i}_stride{j}"
                        for j in range(index_rank)
                    )
                    code.writeline(f"cur_index{i} = tl.load(indices{i}_ptr + {comp})")
                bounds = " & ".join(
                    f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
                    for i in range(indices_len)
                )
                code.writeline(f"valid_slice = {bounds}")
                code.writeline("if not valid_slice:")
                with code.indent():
                    code.writeline("return")
                code.newline()
                comp_in = " + ".join(
                    f"cur_index{i} * input_stride{i}" for i in range(indices_len)
                )
                code.writeline(f"input_slice_base = {comp_in}")
                comp_val = " + ".join(
                    f"indices_idx{j} * values_stride{j}" for j in range(index_rank)
                )
                code.writeline(f"values_slice_base = {comp_val}")
                code.newline()
                code.writeline(
                    "offs1 = pid1 * (BLOCK_SIZE1 * LOOP) + tl.arange(0, BLOCK_SIZE1)"
                )
                code.writeline("for _k in tl.static_range(LOOP):")
                with code.indent():
                    code.writeline("offs = offs1 + _k * BLOCK_SIZE1")
                    code.writeline("if EXACT:")
                    with code.indent():
                        emit_body(None)
                    code.writeline("else:")
                    with code.indent():
                        code.writeline("mask1 = offs < N")
                        emit_body("mask1")

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
    is_1d = inp_rank == indices_len
    code.writeline(
        f"def {wrapper_name}(input, indices, values, accumulate, copy_src=None):"
    )
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("values_shape = values.shape")
        code.writeline("values_stride = values.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()

        if is_1d:
            if index_rank == 1:
                code.writeline("if M >= 32768:")
                with code.indent():
                    code.writeline("if accumulate and input.dtype == torch.float16:")
                    with code.indent():
                        code.writeline("bs0, unroll = 512, 1")
                    code.writeline("elif accumulate:")
                    with code.indent():
                        code.writeline("bs0, unroll = 256, 2")
                    code.writeline("else:")
                    with code.indent():
                        code.writeline("bs0, unroll = 256, 4")
                code.writeline("elif M <= 128:")
                with code.indent():
                    code.writeline(
                        "bs0, unroll = max(32, triton.next_power_of_2(M)), 1"
                    )
                code.writeline("elif M <= 2048:")
                with code.indent():
                    code.writeline("bs0, unroll = 128, 1")
                code.writeline("else:")
                with code.indent():
                    code.writeline("bs0, unroll = 256, 1")
            else:
                code.writeline(
                    "bs0, unroll = max(4, min(128, triton.next_power_of_2(triton.cdiv(M, 120)))), 1"
                )

            code.newline()
            code.writeline("_tile = bs0 * unroll")
            code.writeline(
                "num_warps = 1 if _tile <= 64 else (2 if _tile <= 256 else 4)"
            )
            code.writeline("grid = (triton.cdiv(M, bs0 * unroll),)")
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
                    "accumulate==True,",
                    "BLOCK_SIZE0=bs0,",
                    "UNROLL=unroll,",
                    "num_warps=num_warps,",
                ]
                code.writelines(args)
            code.writeline(")")
        else:
            code.writeline("bs0 = 1")
            code.writeline(
                "is_val_scalar = (values.numel() == 1) or all(st == 0 for st in values_stride)"
            )
            code.writeline(f"trailing_strides = values_stride[{index_rank}:]")
            code.writeline("is_val_contig = False")
            code.writeline(
                "if not is_val_scalar and len(trailing_strides) > 0 and trailing_strides[-1] == 1:"
            )
            with code.indent():
                code.writeline("is_val_contig = True")
                code.writeline("for i in range(len(trailing_strides) - 1, 0, -1):")
                with code.indent():
                    code.writeline(
                        f"if trailing_strides[i - 1] != trailing_strides[i] * input_shape[{indices_len} + i]:"
                    )
                    with code.indent():
                        code.writeline("is_val_contig = False")
                        code.writeline("break")

            code.writeline("is_val_last_dim = False")
            code.writeline(
                "if not is_val_scalar and not is_val_contig and len(trailing_strides) > 1 "
                "and trailing_strides[-1] == 1:"
            )
            with code.indent():
                code.writeline("if all(st == 0 for st in trailing_strides[:-1]):")
                with code.indent():
                    code.writeline("is_val_last_dim = True")

            code.writeline("if is_val_scalar:")
            with code.indent():
                code.writeline("v_mode, v_chunk = 0, 1")
                code.writeline("bs1 = min(2048, max(32, triton.next_power_of_2(N)))")
            code.writeline("elif is_val_contig:")
            with code.indent():
                code.writeline("v_mode, v_chunk = 1, N")
                code.writeline("if N >= 2048: bs1 = 2048")
                code.writeline(
                    "elif N >= 512: bs1 = min(512, triton.next_power_of_2(N))"
                )
                code.writeline(
                    "elif N >= 64: bs1 = min(128, triton.next_power_of_2(N))"
                )
                code.writeline("else: bs1 = max(32, triton.next_power_of_2(N))")
            code.writeline("elif is_val_last_dim:")
            with code.indent():
                code.writeline("v_mode, v_chunk = 2, input_shape[-1]")
                code.writeline("bs1 = min(2048, max(64, triton.next_power_of_2(N)))")
            code.writeline("else:")
            with code.indent():
                code.writeline("v_mode, v_chunk = 3, 1")
                code.writeline("bs1 = min(512, max(32, triton.next_power_of_2(N)))")

            code.newline()
            code.writeline("is_inp_contig = (input_stride[-1] == 1)")
            code.writeline(f"for i in range({inp_rank - 1}, {indices_len}, -1):")
            with code.indent():
                code.writeline(
                    "if input_stride[i - 1] != input_stride[i] * input_shape[i]:"
                )
                with code.indent():
                    code.writeline("is_inp_contig = False")
                    code.writeline("break")

            code.newline()
            code.writeline("block_m = triton.next_power_of_2(max(M, 1))")
            code.writeline("block_n = triton.next_power_of_2(max(N, 1))")
            code.writeline(
                "tiny = (M * N <= 4096) and (block_m <= 512) and (block_n <= 512)"
            )
            code.writeline("loop = 1 if tiny else min(8, max(1, (N // bs1) // 8))")
            code.writeline("chunk = bs1 * loop")
            code.writeline("exact = (not tiny) and (N % chunk == 0)")
            code.writeline(
                "copy_numel = copy_src.numel() if copy_src is not None else 0"
            )
            code.writeline("fuse_copy = tiny and (0 < copy_numel <= 8192)")
            code.writeline(
                "copy_block = max(64, triton.next_power_of_2(copy_numel)) if fuse_copy else 64"
            )
            code.writeline("if copy_src is not None and not fuse_copy:")
            with code.indent():
                code.writeline("input.copy_(copy_src)")
            code.newline()
            code.writeline("es = input.element_size()")
            code.writeline("_tile_bytes = (block_m * block_n if tiny else bs1) * es")
            code.writeline(
                "if fuse_copy: _tile_bytes = max(_tile_bytes, copy_block * es)"
            )
            code.writeline(
                "num_warps = 1 if _tile_bytes <= 128 else (2 if _tile_bytes <= 512 "
                "else (4 if _tile_bytes <= 2048 else 8))"
            )
            code.writeline("use_async = (not tiny) and bs1 >= 512")
            code.writeline("swap = (not tiny) and (M <= 65535) and (N >= 4096)")
            code.writeline("n_split = triton.cdiv(N, chunk)")
            code.writeline(
                "grid = (1,) if tiny else ((n_split, M) if swap else (M, n_split))"
            )
            code.newline()
            code.writeline(f"{kernel_name}[grid](")
            with code.indent():
                args = ["input,"]
                args += [f"indices[{i}]," for i in range(indices_len)]
                args += ["values,"]
                args += ["copy_src if copy_src is not None else input,", "copy_numel,"]
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
                    "accumulate==True,",
                    "BLOCK_SIZE0=bs0,",
                    "BLOCK_SIZE1=bs1,",
                    "IS_INP_CONTIGUOUS=is_inp_contig,",
                    "V_MODE=v_mode,",
                    "V_CHUNK=v_chunk,",
                    "USE_ASYNC=use_async,",
                    "SWAP_GRID=swap,",
                    "LOOP=loop,",
                    "EXACT=exact,",
                    "TINY=tiny,",
                    "BLOCK_M=block_m,",
                    "BLOCK_N=block_n,",
                    "FUSE_COPY=fuse_copy,",
                    "COPY_BLOCK=copy_block,",
                    "num_warps=num_warps,",
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
        inp, tensor_indices, values, accumulate = args[:4]
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
    if (
        not accumulate
        and len(indices) == 1
        and indices[0] is not None
        and indices[0].dtype == torch.bool
    ):
        mask = indices[0]
        if mask.shape == inp.shape:
            out = inp.clone()
            if mask.device != inp.device:
                mask = mask.to(inp.device)
            if values.device != inp.device:
                values = values.to(inp.device)
            out.masked_scatter_(mask, values)
            return out

    if (
        inp.numel() <= 8192
        and len(indices) == 1
        and indices[0] is not None
        and indices[0].dtype in (torch.int64, torch.int32)
    ):
        out = torch.empty_like(inp)
        return index_put_(out, indices, values, accumulate, inp)

    out = inp.clone()
    return index_put_(out, indices, values, accumulate)


def index_put_(inp, indices, values, accumulate=False, copy_src=None):
    if not indices:
        raise ValueError("At least one index tensor is required")

    inp_device = inp.device

    if (
        not accumulate
        and len(indices) == 1
        and indices[0] is not None
        and indices[0].dtype == torch.bool
    ):
        mask = indices[0]
        if mask.shape == inp.shape:
            if mask.device != inp_device:
                mask = mask.to(inp_device)
            if values.device != inp_device:
                values = values.to(inp_device)
            inp.masked_scatter_(mask, values)
            return inp

    if (
        len(indices) == 1
        and indices[0] is not None
        and indices[0].dtype in (torch.int64, torch.int32)
    ):
        idx = indices[0]
        if idx.device != inp_device:
            idx = idx.to(inp_device)
        if values.device != inp_device:
            values = values.to(inp_device)

        target_shape = tuple(idx.shape) + tuple(inp.shape[1:])
        if tuple(values.shape) != target_shape:
            values = values.broadcast_to(target_shape)

        return _index_put_func(inp, [idx], values, accumulate, copy_src)

    indices = list(indices)
    indices = [
        (
            index.to(inp_device)
            if index is not None and index.device != inp_device
            else index
        )
        for index in indices
    ]

    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.int8):
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif torch.is_tensor(idx):
            processed_indices.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    indices = processed_indices
    inp_ndim = inp.ndim
    if len(indices) < inp_ndim:
        indices.extend([None] * (inp_ndim - len(indices)))
    elif len(indices) > inp_ndim:
        raise IndexError(f"too many indices for tensor of dimension {inp_ndim}")

    tensor_pos = [i for i, x in enumerate(indices) if x is not None]
    if not tensor_pos:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [indices[i] for i in tensor_pos]
    if len(tensor_indices) > 1:
        broadcasted = torch.broadcast_tensors(*tensor_indices)
        for i, pos in enumerate(tensor_pos):
            indices[pos] = broadcasted[i]

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

    tensors = [x for x in final_indices if x is not None]
    broadcast_shape = list(tensors[0].shape)
    slice_shape = [inp_view.shape[i] for i, x in enumerate(final_indices) if x is None]
    target_shape = broadcast_shape + slice_shape

    if values.device != inp_device:
        values = values.to(inp_device)

    if need_transpose and is_contiguous:
        num_before = tensor_pos[0]
        before_dims = slice_shape[:num_before]
        after_dims = slice_shape[num_before:]
        natural_shape = before_dims + broadcast_shape + after_dims
        values = values.broadcast_to(natural_shape)
        B, T = len(before_dims), len(broadcast_shape)
        val_perm = (
            list(range(B, B + T)) + list(range(0, B)) + list(range(B + T, values.ndim))
        )
        values = values.permute(val_perm)
    else:
        if tuple(values.shape) != tuple(target_shape):
            values = values.broadcast_to(target_shape)

    _index_put_func(inp_view, tensors, values, accumulate, copy_src)
    return inp


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    return index_put_(inp, indices, values, accumulate)
