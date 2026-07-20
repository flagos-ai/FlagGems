# Copyright 2026, The FlagOS Contributors.
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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_cross import (
    _get_strided_layout,
    _prepare_inputs,
    _resolve_view,
    _validate_inputs,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = (torch.float32, torch.complex64)


@libentry()
@triton.jit
def _linalg_cross_real_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    BLOCK_SIZE: tl.constexpr,
):
    local_offsets = tl.arange(0, BLOCK_SIZE * 4)
    valid_local = local_offsets < BLOCK_SIZE * 3
    block_start = ext.program_id(0) * BLOCK_SIZE * 3
    global_offsets = block_start + local_offsets
    num_elements = num_vectors * 3
    mask = valid_local & (global_offsets < num_elements)

    input_values = tl.load(input_ptr + global_offsets, mask=mask, other=0.0)
    other_values = tl.load(other_ptr + global_offsets, mask=mask, other=0.0)

    safe_offsets = tl.where(valid_local, local_offsets, 0)
    component = safe_offsets % 3
    vector_base = safe_offsets - component
    next_component = tl.where(component == 2, 0, component + 1)
    last_component = tl.where(component == 0, 2, component - 1)
    next_offsets = vector_base + next_component
    last_offsets = vector_base + last_component
    input_next = tl.gather(input_values, next_offsets, axis=0)
    input_last = tl.gather(input_values, last_offsets, axis=0)
    other_next = tl.gather(other_values, next_offsets, axis=0)
    other_last = tl.gather(other_values, last_offsets, axis=0)

    output_values = input_next * other_last - input_last * other_next
    tl.store(output_ptr + global_offsets, output_values, mask=mask)


@libentry()
@triton.jit
def _linalg_cross_complex_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    BLOCK_SIZE: tl.constexpr,
):
    vector_offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vector_offsets < num_vectors
    offsets = vector_offsets * 6

    input_0_real = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    input_0_imag = tl.load(input_ptr + offsets + 1, mask=mask, other=0.0)
    input_1_real = tl.load(input_ptr + offsets + 2, mask=mask, other=0.0)
    input_1_imag = tl.load(input_ptr + offsets + 3, mask=mask, other=0.0)
    input_2_real = tl.load(input_ptr + offsets + 4, mask=mask, other=0.0)
    input_2_imag = tl.load(input_ptr + offsets + 5, mask=mask, other=0.0)

    other_0_real = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    other_0_imag = tl.load(other_ptr + offsets + 1, mask=mask, other=0.0)
    other_1_real = tl.load(other_ptr + offsets + 2, mask=mask, other=0.0)
    other_1_imag = tl.load(other_ptr + offsets + 3, mask=mask, other=0.0)
    other_2_real = tl.load(other_ptr + offsets + 4, mask=mask, other=0.0)
    other_2_imag = tl.load(other_ptr + offsets + 5, mask=mask, other=0.0)

    product_12_real = input_1_real * other_2_real - input_1_imag * other_2_imag
    product_12_imag = input_1_real * other_2_imag + input_1_imag * other_2_real
    product_21_real = input_2_real * other_1_real - input_2_imag * other_1_imag
    product_21_imag = input_2_real * other_1_imag + input_2_imag * other_1_real

    product_20_real = input_2_real * other_0_real - input_2_imag * other_0_imag
    product_20_imag = input_2_real * other_0_imag + input_2_imag * other_0_real
    product_02_real = input_0_real * other_2_real - input_0_imag * other_2_imag
    product_02_imag = input_0_real * other_2_imag + input_0_imag * other_2_real

    product_01_real = input_0_real * other_1_real - input_0_imag * other_1_imag
    product_01_imag = input_0_real * other_1_imag + input_0_imag * other_1_real
    product_10_real = input_1_real * other_0_real - input_1_imag * other_0_imag
    product_10_imag = input_1_real * other_0_imag + input_1_imag * other_0_real

    tl.store(output_ptr + offsets, product_12_real - product_21_real, mask=mask)
    tl.store(output_ptr + offsets + 1, product_12_imag - product_21_imag, mask=mask)
    tl.store(output_ptr + offsets + 2, product_20_real - product_02_real, mask=mask)
    tl.store(output_ptr + offsets + 3, product_20_imag - product_02_imag, mask=mask)
    tl.store(output_ptr + offsets + 4, product_01_real - product_10_real, mask=mask)
    tl.store(output_ptr + offsets + 5, product_01_imag - product_10_imag, mask=mask)


@libentry()
@triton.jit
def _linalg_cross_complex_contiguous_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    BLOCK_SIZE: tl.constexpr,
):
    vector_offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    columns = tl.arange(0, 8)
    vector_mask = vector_offsets < num_vectors
    offsets = vector_offsets[:, None] * 6 + columns[None, :]
    mask = vector_mask[:, None] & (columns[None, :] < 6)

    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_values = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    input_real, input_imag = tl.split(
        tl.reshape(input_values, (BLOCK_SIZE * 4, 2))
    )
    input_real_even, input_real_odd = tl.split(
        tl.reshape(input_real, (BLOCK_SIZE * 2, 2))
    )
    input_imag_even, input_imag_odd = tl.split(
        tl.reshape(input_imag, (BLOCK_SIZE * 2, 2))
    )
    input_0_real, input_2_real = tl.split(
        tl.reshape(input_real_even, (BLOCK_SIZE, 2))
    )
    input_1_real, _ = tl.split(tl.reshape(input_real_odd, (BLOCK_SIZE, 2)))
    input_0_imag, input_2_imag = tl.split(
        tl.reshape(input_imag_even, (BLOCK_SIZE, 2))
    )
    input_1_imag, _ = tl.split(tl.reshape(input_imag_odd, (BLOCK_SIZE, 2)))

    other_real, other_imag = tl.split(
        tl.reshape(other_values, (BLOCK_SIZE * 4, 2))
    )
    other_real_even, other_real_odd = tl.split(
        tl.reshape(other_real, (BLOCK_SIZE * 2, 2))
    )
    other_imag_even, other_imag_odd = tl.split(
        tl.reshape(other_imag, (BLOCK_SIZE * 2, 2))
    )
    other_0_real, other_2_real = tl.split(
        tl.reshape(other_real_even, (BLOCK_SIZE, 2))
    )
    other_1_real, _ = tl.split(tl.reshape(other_real_odd, (BLOCK_SIZE, 2)))
    other_0_imag, other_2_imag = tl.split(
        tl.reshape(other_imag_even, (BLOCK_SIZE, 2))
    )
    other_1_imag, _ = tl.split(tl.reshape(other_imag_odd, (BLOCK_SIZE, 2)))

    out_0_real = (
        input_1_real * other_2_real
        - input_1_imag * other_2_imag
        - input_2_real * other_1_real
        + input_2_imag * other_1_imag
    )
    out_0_imag = (
        input_1_real * other_2_imag
        + input_1_imag * other_2_real
        - input_2_real * other_1_imag
        - input_2_imag * other_1_real
    )
    out_1_real = (
        input_2_real * other_0_real
        - input_2_imag * other_0_imag
        - input_0_real * other_2_real
        + input_0_imag * other_2_imag
    )
    out_1_imag = (
        input_2_real * other_0_imag
        + input_2_imag * other_0_real
        - input_0_real * other_2_imag
        - input_0_imag * other_2_real
    )
    out_2_real = (
        input_0_real * other_1_real
        - input_0_imag * other_1_imag
        - input_1_real * other_0_real
        + input_1_imag * other_0_imag
    )
    out_2_imag = (
        input_0_real * other_1_imag
        + input_0_imag * other_1_real
        - input_1_real * other_0_imag
        - input_1_imag * other_0_real
    )

    output_values = tl.where(
        columns[None, :] == 0,
        out_0_real[:, None],
        tl.where(
            columns[None, :] == 1,
            out_0_imag[:, None],
            tl.where(
                columns[None, :] == 2,
                out_1_real[:, None],
                tl.where(
                    columns[None, :] == 3,
                    out_1_imag[:, None],
                    tl.where(
                        columns[None, :] == 4,
                        out_2_real[:, None],
                        out_2_imag[:, None],
                    ),
                ),
            ),
        ),
    )
    tl.store(output_ptr + offsets, output_values, mask=mask)


@libentry()
@triton.jit
def _linalg_cross_real_strided_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    INPUT_OUTER0_STRIDE: tl.constexpr,
    INPUT_OUTER1_STRIDE: tl.constexpr,
    INPUT_COMPONENT_STRIDE: tl.constexpr,
    OTHER_OUTER0_STRIDE: tl.constexpr,
    OTHER_OUTER1_STRIDE: tl.constexpr,
    OTHER_COMPONENT_STRIDE: tl.constexpr,
    OUTPUT_OUTER0_STRIDE: tl.constexpr,
    OUTPUT_OUTER1_STRIDE: tl.constexpr,
    OUTPUT_COMPONENT_STRIDE: tl.constexpr,
    OUTER1_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vector_offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vector_offsets < num_vectors
    outer0 = vector_offsets // OUTER1_SIZE
    outer1 = vector_offsets % OUTER1_SIZE

    input_base = outer0 * INPUT_OUTER0_STRIDE + outer1 * INPUT_OUTER1_STRIDE
    other_base = outer0 * OTHER_OUTER0_STRIDE + outer1 * OTHER_OUTER1_STRIDE
    output_base = outer0 * OUTPUT_OUTER0_STRIDE + outer1 * OUTPUT_OUTER1_STRIDE

    input_0 = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    input_1 = tl.load(
        input_ptr + input_base + INPUT_COMPONENT_STRIDE, mask=mask, other=0.0
    )
    input_2 = tl.load(
        input_ptr + input_base + 2 * INPUT_COMPONENT_STRIDE, mask=mask, other=0.0
    )
    other_0 = tl.load(other_ptr + other_base, mask=mask, other=0.0)
    other_1 = tl.load(
        other_ptr + other_base + OTHER_COMPONENT_STRIDE, mask=mask, other=0.0
    )
    other_2 = tl.load(
        other_ptr + other_base + 2 * OTHER_COMPONENT_STRIDE, mask=mask, other=0.0
    )

    tl.store(
        output_ptr + output_base,
        input_1 * other_2 - input_2 * other_1,
        mask=mask,
    )
    tl.store(
        output_ptr + output_base + OUTPUT_COMPONENT_STRIDE,
        input_2 * other_0 - input_0 * other_2,
        mask=mask,
    )
    tl.store(
        output_ptr + output_base + 2 * OUTPUT_COMPONENT_STRIDE,
        input_0 * other_1 - input_1 * other_0,
        mask=mask,
    )


@libentry()
@triton.jit
def _linalg_cross_complex_strided_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    INPUT_OUTER0_STRIDE: tl.constexpr,
    INPUT_OUTER1_STRIDE: tl.constexpr,
    INPUT_COMPONENT_STRIDE: tl.constexpr,
    OTHER_OUTER0_STRIDE: tl.constexpr,
    OTHER_OUTER1_STRIDE: tl.constexpr,
    OTHER_COMPONENT_STRIDE: tl.constexpr,
    OUTPUT_OUTER0_STRIDE: tl.constexpr,
    OUTPUT_OUTER1_STRIDE: tl.constexpr,
    OUTPUT_COMPONENT_STRIDE: tl.constexpr,
    OUTER1_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vector_offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vector_offsets < num_vectors
    outer0 = vector_offsets // OUTER1_SIZE
    outer1 = vector_offsets % OUTER1_SIZE

    input_base = outer0 * INPUT_OUTER0_STRIDE + outer1 * INPUT_OUTER1_STRIDE
    other_base = outer0 * OTHER_OUTER0_STRIDE + outer1 * OTHER_OUTER1_STRIDE
    output_base = outer0 * OUTPUT_OUTER0_STRIDE + outer1 * OUTPUT_OUTER1_STRIDE

    input_0_real = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    input_0_imag = tl.load(input_ptr + input_base + 1, mask=mask, other=0.0)
    input_1_real = tl.load(
        input_ptr + input_base + INPUT_COMPONENT_STRIDE, mask=mask, other=0.0
    )
    input_1_imag = tl.load(
        input_ptr + input_base + INPUT_COMPONENT_STRIDE + 1,
        mask=mask,
        other=0.0,
    )
    input_2_real = tl.load(
        input_ptr + input_base + 2 * INPUT_COMPONENT_STRIDE,
        mask=mask,
        other=0.0,
    )
    input_2_imag = tl.load(
        input_ptr + input_base + 2 * INPUT_COMPONENT_STRIDE + 1,
        mask=mask,
        other=0.0,
    )

    other_0_real = tl.load(other_ptr + other_base, mask=mask, other=0.0)
    other_0_imag = tl.load(other_ptr + other_base + 1, mask=mask, other=0.0)
    other_1_real = tl.load(
        other_ptr + other_base + OTHER_COMPONENT_STRIDE, mask=mask, other=0.0
    )
    other_1_imag = tl.load(
        other_ptr + other_base + OTHER_COMPONENT_STRIDE + 1,
        mask=mask,
        other=0.0,
    )
    other_2_real = tl.load(
        other_ptr + other_base + 2 * OTHER_COMPONENT_STRIDE,
        mask=mask,
        other=0.0,
    )
    other_2_imag = tl.load(
        other_ptr + other_base + 2 * OTHER_COMPONENT_STRIDE + 1,
        mask=mask,
        other=0.0,
    )

    out_0_real = (
        input_1_real * other_2_real
        - input_1_imag * other_2_imag
        - input_2_real * other_1_real
        + input_2_imag * other_1_imag
    )
    out_0_imag = (
        input_1_real * other_2_imag
        + input_1_imag * other_2_real
        - input_2_real * other_1_imag
        - input_2_imag * other_1_real
    )
    out_1_real = (
        input_2_real * other_0_real
        - input_2_imag * other_0_imag
        - input_0_real * other_2_real
        + input_0_imag * other_2_imag
    )
    out_1_imag = (
        input_2_real * other_0_imag
        + input_2_imag * other_0_real
        - input_0_real * other_2_imag
        - input_0_imag * other_2_real
    )
    out_2_real = (
        input_0_real * other_1_real
        - input_0_imag * other_1_imag
        - input_1_real * other_0_real
        + input_1_imag * other_0_imag
    )
    out_2_imag = (
        input_0_real * other_1_imag
        + input_0_imag * other_1_real
        - input_1_real * other_0_imag
        - input_1_imag * other_0_real
    )

    tl.store(output_ptr + output_base, out_0_real, mask=mask)
    tl.store(output_ptr + output_base + 1, out_0_imag, mask=mask)
    tl.store(
        output_ptr + output_base + OUTPUT_COMPONENT_STRIDE, out_1_real, mask=mask
    )
    tl.store(
        output_ptr + output_base + OUTPUT_COMPONENT_STRIDE + 1,
        out_1_imag,
        mask=mask,
    )
    tl.store(
        output_ptr + output_base + 2 * OUTPUT_COMPONENT_STRIDE,
        out_2_real,
        mask=mask,
    )
    tl.store(
        output_ptr + output_base + 2 * OUTPUT_COMPONENT_STRIDE + 1,
        out_2_imag,
        mask=mask,
    )


@libentry()
@triton.jit
def _linalg_cross_real_lastdim_broadcast_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    INPUT_VECTORS: tl.constexpr,
    OTHER_VECTORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    local_offsets = tl.arange(0, BLOCK_SIZE * 4)
    valid_local = local_offsets < BLOCK_SIZE * 3
    block_start = ext.program_id(0) * BLOCK_SIZE * 3
    global_offsets = block_start + local_offsets
    mask = valid_local & (global_offsets < num_vectors * 3)
    safe_offsets = tl.where(valid_local, local_offsets, 0)
    component = safe_offsets % 3

    if INPUT_VECTORS == 1:
        input_0 = tl.load(input_ptr)
        input_1 = tl.load(input_ptr + 1)
        input_2 = tl.load(input_ptr + 2)
        input_values = tl.where(
            component == 0, input_0, tl.where(component == 1, input_1, input_2)
        )
    else:
        input_values = tl.load(input_ptr + global_offsets, mask=mask, other=0.0)
    if OTHER_VECTORS == 1:
        other_0 = tl.load(other_ptr)
        other_1 = tl.load(other_ptr + 1)
        other_2 = tl.load(other_ptr + 2)
        other_values = tl.where(
            component == 0, other_0, tl.where(component == 1, other_1, other_2)
        )
    else:
        other_values = tl.load(other_ptr + global_offsets, mask=mask, other=0.0)

    vector_base = safe_offsets - component
    next_component = tl.where(component == 2, 0, component + 1)
    last_component = tl.where(component == 0, 2, component - 1)
    next_offsets = vector_base + next_component
    last_offsets = vector_base + last_component
    input_next = tl.gather(input_values, next_offsets, axis=0)
    input_last = tl.gather(input_values, last_offsets, axis=0)
    other_next = tl.gather(other_values, next_offsets, axis=0)
    other_last = tl.gather(other_values, last_offsets, axis=0)

    output_values = input_next * other_last - input_last * other_next
    tl.store(output_ptr + global_offsets, output_values, mask=mask)


@libentry()
@triton.jit
def _linalg_cross_complex_lastdim_broadcast_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    INPUT_VECTORS: tl.constexpr,
    OTHER_VECTORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vectors = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vectors < num_vectors
    output_base = vectors * 6

    if INPUT_VECTORS == 1:
        input_0_real = tl.load(input_ptr)
        input_0_imag = tl.load(input_ptr + 1)
        input_1_real = tl.load(input_ptr + 2)
        input_1_imag = tl.load(input_ptr + 3)
        input_2_real = tl.load(input_ptr + 4)
        input_2_imag = tl.load(input_ptr + 5)
    else:
        input_base = vectors * 6
        input_0_real = tl.load(input_ptr + input_base, mask=mask, other=0.0)
        input_0_imag = tl.load(input_ptr + input_base + 1, mask=mask, other=0.0)
        input_1_real = tl.load(input_ptr + input_base + 2, mask=mask, other=0.0)
        input_1_imag = tl.load(input_ptr + input_base + 3, mask=mask, other=0.0)
        input_2_real = tl.load(input_ptr + input_base + 4, mask=mask, other=0.0)
        input_2_imag = tl.load(input_ptr + input_base + 5, mask=mask, other=0.0)
    if OTHER_VECTORS == 1:
        other_0_real = tl.load(other_ptr)
        other_0_imag = tl.load(other_ptr + 1)
        other_1_real = tl.load(other_ptr + 2)
        other_1_imag = tl.load(other_ptr + 3)
        other_2_real = tl.load(other_ptr + 4)
        other_2_imag = tl.load(other_ptr + 5)
    else:
        other_base = vectors * 6
        other_0_real = tl.load(other_ptr + other_base, mask=mask, other=0.0)
        other_0_imag = tl.load(other_ptr + other_base + 1, mask=mask, other=0.0)
        other_1_real = tl.load(other_ptr + other_base + 2, mask=mask, other=0.0)
        other_1_imag = tl.load(other_ptr + other_base + 3, mask=mask, other=0.0)
        other_2_real = tl.load(other_ptr + other_base + 4, mask=mask, other=0.0)
        other_2_imag = tl.load(other_ptr + other_base + 5, mask=mask, other=0.0)

    out_0_real = (
        input_1_real * other_2_real
        - input_1_imag * other_2_imag
        - input_2_real * other_1_real
        + input_2_imag * other_1_imag
    )
    out_0_imag = (
        input_1_real * other_2_imag
        + input_1_imag * other_2_real
        - input_2_real * other_1_imag
        - input_2_imag * other_1_real
    )
    out_1_real = (
        input_2_real * other_0_real
        - input_2_imag * other_0_imag
        - input_0_real * other_2_real
        + input_0_imag * other_2_imag
    )
    out_1_imag = (
        input_2_real * other_0_imag
        + input_2_imag * other_0_real
        - input_0_real * other_2_imag
        - input_0_imag * other_2_real
    )
    out_2_real = (
        input_0_real * other_1_real
        - input_0_imag * other_1_imag
        - input_1_real * other_0_real
        + input_1_imag * other_0_imag
    )
    out_2_imag = (
        input_0_real * other_1_imag
        + input_0_imag * other_1_real
        - input_1_real * other_0_imag
        - input_1_imag * other_0_real
    )
    tl.store(output_ptr + output_base, out_0_real, mask=mask)
    tl.store(output_ptr + output_base + 1, out_0_imag, mask=mask)
    tl.store(output_ptr + output_base + 2, out_1_real, mask=mask)
    tl.store(output_ptr + output_base + 3, out_1_imag, mask=mask)
    tl.store(output_ptr + output_base + 4, out_2_real, mask=mask)
    tl.store(output_ptr + output_base + 5, out_2_imag, mask=mask)


@libentry()
@triton.jit
def _linalg_cross_real_dim1_3d_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    INNER_SIZE: tl.constexpr,
    INPUT_BATCH_STRIDE: tl.constexpr,
    OTHER_BATCH_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    inner = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = inner < INNER_SIZE
    input_base = batch * INPUT_BATCH_STRIDE + inner
    other_base = batch * OTHER_BATCH_STRIDE + inner
    output_base = batch * (3 * INNER_SIZE) + inner

    input_0 = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    input_1 = tl.load(input_ptr + input_base + INNER_SIZE, mask=mask, other=0.0)
    input_2 = tl.load(input_ptr + input_base + 2 * INNER_SIZE, mask=mask, other=0.0)
    other_0 = tl.load(other_ptr + other_base, mask=mask, other=0.0)
    other_1 = tl.load(other_ptr + other_base + INNER_SIZE, mask=mask, other=0.0)
    other_2 = tl.load(other_ptr + other_base + 2 * INNER_SIZE, mask=mask, other=0.0)

    tl.store(output_ptr + output_base, input_1 * other_2 - input_2 * other_1, mask=mask)
    tl.store(
        output_ptr + output_base + INNER_SIZE,
        input_2 * other_0 - input_0 * other_2,
        mask=mask,
    )
    tl.store(
        output_ptr + output_base + 2 * INNER_SIZE,
        input_0 * other_1 - input_1 * other_0,
        mask=mask,
    )


@libentry()
@triton.jit
def _linalg_cross_real_dim1_small_inner_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_batches,
    INNER_SIZE: tl.constexpr,
    INPUT_BATCH_STRIDE: tl.constexpr,
    OTHER_BATCH_STRIDE: tl.constexpr,
    BATCH_BLOCK_SIZE: tl.constexpr,
    INNER_BLOCK_SIZE: tl.constexpr,
):
    batches = tl.program_id(0) * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)
    inner = tl.arange(0, INNER_BLOCK_SIZE)
    batch_mask = batches[:, None] < num_batches
    inner_mask = inner[None, :] < INNER_SIZE

    if INPUT_BATCH_STRIDE == 0:
        input_base = inner[None, :]
        input_mask = inner_mask
    else:
        input_base = batches[:, None] * INPUT_BATCH_STRIDE + inner[None, :]
        input_mask = batch_mask & inner_mask
    if OTHER_BATCH_STRIDE == 0:
        other_base = inner[None, :]
        other_mask = inner_mask
    else:
        other_base = batches[:, None] * OTHER_BATCH_STRIDE + inner[None, :]
        other_mask = batch_mask & inner_mask

    input_0 = tl.load(input_ptr + input_base, mask=input_mask, other=0.0)
    input_1 = tl.load(
        input_ptr + input_base + INNER_SIZE, mask=input_mask, other=0.0
    )
    input_2 = tl.load(
        input_ptr + input_base + 2 * INNER_SIZE, mask=input_mask, other=0.0
    )
    other_0 = tl.load(other_ptr + other_base, mask=other_mask, other=0.0)
    other_1 = tl.load(
        other_ptr + other_base + INNER_SIZE, mask=other_mask, other=0.0
    )
    other_2 = tl.load(
        other_ptr + other_base + 2 * INNER_SIZE, mask=other_mask, other=0.0
    )

    output_base = batches[:, None] * (3 * INNER_SIZE) + inner[None, :]
    output_mask = batch_mask & inner_mask
    tl.store(
        output_ptr + output_base,
        input_1 * other_2 - input_2 * other_1,
        mask=output_mask,
    )
    tl.store(
        output_ptr + output_base + INNER_SIZE,
        input_2 * other_0 - input_0 * other_2,
        mask=output_mask,
    )
    tl.store(
        output_ptr + output_base + 2 * INNER_SIZE,
        input_0 * other_1 - input_1 * other_0,
        mask=output_mask,
    )


@libentry()
@triton.jit
def _linalg_cross_complex_dim1_3d_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    INNER_SIZE: tl.constexpr,
    INPUT_BATCH_STRIDE: tl.constexpr,
    OTHER_BATCH_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    inner = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = inner < INNER_SIZE
    input_base = 2 * (batch * INPUT_BATCH_STRIDE + inner)
    other_base = 2 * (batch * OTHER_BATCH_STRIDE + inner)
    output_base = 2 * (batch * (3 * INNER_SIZE) + inner)
    component_stride = 2 * INNER_SIZE

    input_0_real = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    input_0_imag = tl.load(input_ptr + input_base + 1, mask=mask, other=0.0)
    input_1_real = tl.load(
        input_ptr + input_base + component_stride, mask=mask, other=0.0
    )
    input_1_imag = tl.load(
        input_ptr + input_base + component_stride + 1, mask=mask, other=0.0
    )
    input_2_real = tl.load(
        input_ptr + input_base + 2 * component_stride, mask=mask, other=0.0
    )
    input_2_imag = tl.load(
        input_ptr + input_base + 2 * component_stride + 1, mask=mask, other=0.0
    )
    other_0_real = tl.load(other_ptr + other_base, mask=mask, other=0.0)
    other_0_imag = tl.load(other_ptr + other_base + 1, mask=mask, other=0.0)
    other_1_real = tl.load(
        other_ptr + other_base + component_stride, mask=mask, other=0.0
    )
    other_1_imag = tl.load(
        other_ptr + other_base + component_stride + 1, mask=mask, other=0.0
    )
    other_2_real = tl.load(
        other_ptr + other_base + 2 * component_stride, mask=mask, other=0.0
    )
    other_2_imag = tl.load(
        other_ptr + other_base + 2 * component_stride + 1, mask=mask, other=0.0
    )

    out_0_real = (
        input_1_real * other_2_real
        - input_1_imag * other_2_imag
        - input_2_real * other_1_real
        + input_2_imag * other_1_imag
    )
    out_0_imag = (
        input_1_real * other_2_imag
        + input_1_imag * other_2_real
        - input_2_real * other_1_imag
        - input_2_imag * other_1_real
    )
    out_1_real = (
        input_2_real * other_0_real
        - input_2_imag * other_0_imag
        - input_0_real * other_2_real
        + input_0_imag * other_2_imag
    )
    out_1_imag = (
        input_2_real * other_0_imag
        + input_2_imag * other_0_real
        - input_0_real * other_2_imag
        - input_0_imag * other_2_real
    )
    out_2_real = (
        input_0_real * other_1_real
        - input_0_imag * other_1_imag
        - input_1_real * other_0_real
        + input_1_imag * other_0_imag
    )
    out_2_imag = (
        input_0_real * other_1_imag
        + input_0_imag * other_1_real
        - input_1_real * other_0_imag
        - input_1_imag * other_0_real
    )

    tl.store(output_ptr + output_base, out_0_real, mask=mask)
    tl.store(output_ptr + output_base + 1, out_0_imag, mask=mask)
    tl.store(output_ptr + output_base + component_stride, out_1_real, mask=mask)
    tl.store(output_ptr + output_base + component_stride + 1, out_1_imag, mask=mask)
    tl.store(output_ptr + output_base + 2 * component_stride, out_2_real, mask=mask)
    tl.store(
        output_ptr + output_base + 2 * component_stride + 1,
        out_2_imag,
        mask=mask,
    )


def _launch_linalg_cross_kernel(input, other, output, dim, layout=None):
    num_vectors = output.numel() // 3
    if num_vectors == 0:
        return

    block_size = 128 if output.is_complex() else 256
    grid = (triton.cdiv(num_vectors, block_size),)
    with torch_device_fn.device(output.device):
        if output.is_complex():
            input_real = torch.view_as_real(input)
            other_real = torch.view_as_real(other)
            output_real = torch.view_as_real(output)
            if layout is None:
                _linalg_cross_complex_kernel[grid](
                    input_real,
                    other_real,
                    output_real,
                    num_vectors,
                    BLOCK_SIZE=block_size,
                )
            else:
                (
                    input_outer,
                    other_outer,
                    output_outer,
                    input_component,
                    other_component,
                    output_component,
                    outer1_size,
                ) = layout
                _linalg_cross_complex_strided_kernel[grid](
                    input_real,
                    other_real,
                    output_real,
                    num_vectors,
                    INPUT_OUTER0_STRIDE=2 * input_outer[0],
                    INPUT_OUTER1_STRIDE=2 * input_outer[1],
                    INPUT_COMPONENT_STRIDE=2 * input_component,
                    OTHER_OUTER0_STRIDE=2 * other_outer[0],
                    OTHER_OUTER1_STRIDE=2 * other_outer[1],
                    OTHER_COMPONENT_STRIDE=2 * other_component,
                    OUTPUT_OUTER0_STRIDE=2 * output_outer[0],
                    OUTPUT_OUTER1_STRIDE=2 * output_outer[1],
                    OUTPUT_COMPONENT_STRIDE=2 * output_component,
                    OUTER1_SIZE=outer1_size,
                    BLOCK_SIZE=block_size,
                )
        else:
            if layout is None:
                _linalg_cross_real_kernel[grid](
                    input,
                    other,
                    output,
                    num_vectors,
                    BLOCK_SIZE=block_size,
                )
            else:
                (
                    input_outer,
                    other_outer,
                    output_outer,
                    input_component,
                    other_component,
                    output_component,
                    outer1_size,
                ) = layout
                _linalg_cross_real_strided_kernel[grid](
                    input,
                    other,
                    output,
                    num_vectors,
                    INPUT_OUTER0_STRIDE=input_outer[0],
                    INPUT_OUTER1_STRIDE=input_outer[1],
                    INPUT_COMPONENT_STRIDE=input_component,
                    OTHER_OUTER0_STRIDE=other_outer[0],
                    OTHER_OUTER1_STRIDE=other_outer[1],
                    OTHER_COMPONENT_STRIDE=other_component,
                    OUTPUT_OUTER0_STRIDE=output_outer[0],
                    OUTPUT_OUTER1_STRIDE=output_outer[1],
                    OUTPUT_COMPONENT_STRIDE=output_component,
                    OUTER1_SIZE=outer1_size,
                    BLOCK_SIZE=block_size,
                )


def _launch_lastdim_broadcast_kernel(input, other, output):
    num_vectors = output.numel() // 3
    block_size = 128 if output.is_complex() else 256
    grid = (triton.cdiv(num_vectors, block_size),)
    with torch_device_fn.device(output.device):
        if output.is_complex():
            _linalg_cross_complex_lastdim_broadcast_kernel[grid](
                torch.view_as_real(input),
                torch.view_as_real(other),
                torch.view_as_real(output),
                num_vectors,
                INPUT_VECTORS=input.numel() // 3,
                OTHER_VECTORS=other.numel() // 3,
                BLOCK_SIZE=block_size,
            )
        else:
            _linalg_cross_real_lastdim_broadcast_kernel[grid](
                input,
                other,
                output,
                num_vectors,
                INPUT_VECTORS=input.numel() // 3,
                OTHER_VECTORS=other.numel() // 3,
                BLOCK_SIZE=block_size,
            )


def _launch_dim1_3d_kernel(input, other, output):
    block_size = 128 if output.is_complex() else 256
    grid = (output.shape[0], triton.cdiv(output.shape[2], block_size))
    input_batch_stride = 0 if input.shape[0] == 1 else input.stride(0)
    other_batch_stride = 0 if other.shape[0] == 1 else other.stride(0)
    with torch_device_fn.device(output.device):
        if output.is_complex():
            _linalg_cross_complex_dim1_3d_kernel[grid](
                torch.view_as_real(input),
                torch.view_as_real(other),
                torch.view_as_real(output),
                INNER_SIZE=output.shape[2],
                INPUT_BATCH_STRIDE=input_batch_stride,
                OTHER_BATCH_STRIDE=other_batch_stride,
                BLOCK_SIZE=block_size,
            )
        else:
            _linalg_cross_real_dim1_3d_kernel[grid](
                input,
                other,
                output,
                INNER_SIZE=output.shape[2],
                INPUT_BATCH_STRIDE=input_batch_stride,
                OTHER_BATCH_STRIDE=other_batch_stride,
                BLOCK_SIZE=block_size,
            )


def _launch_dim1_small_inner_kernel(input, other, output):
    batch_block_size = min(64, triton.next_power_of_2(output.shape[0]))
    inner_block_size = triton.next_power_of_2(output.shape[2])
    grid = (triton.cdiv(output.shape[0], batch_block_size),)
    input_batch_stride = 0 if input.shape[0] == 1 else input.stride(0)
    other_batch_stride = 0 if other.shape[0] == 1 else other.stride(0)
    with torch_device_fn.device(output.device):
        _linalg_cross_real_dim1_small_inner_kernel[grid](
            input,
            other,
            output,
            output.shape[0],
            INNER_SIZE=output.shape[2],
            INPUT_BATCH_STRIDE=input_batch_stride,
            OTHER_BATCH_STRIDE=other_batch_stride,
            BATCH_BLOCK_SIZE=batch_block_size,
            INNER_BLOCK_SIZE=inner_block_size,
        )


def _linalg_cross_impl(input, other, dim):
    if input.dtype not in _SUPPORTED_DTYPES or other.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "linalg_cross on Ascend only supports float32 and complex64"
        )
    dim, output_shape = _validate_inputs(input, other, dim)
    input = _resolve_view(input)
    other = _resolve_view(other)

    if input.ndim <= 3:
        fast_contiguous = (
            dim == input.ndim - 1
            and input.shape == output_shape
            and other.shape == output_shape
            and input.is_contiguous()
            and other.is_contiguous()
        )
        if fast_contiguous:
            output = torch.empty_like(input)
            _launch_linalg_cross_kernel(input, other, output, dim)
            return output

        if input.shape == output_shape and input.is_contiguous():
            output = torch.empty_like(input)
        elif other.shape == output_shape and other.is_contiguous():
            output = torch.empty_like(other)
        else:
            output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

        if (
            input.ndim == 2
            and dim == 1
            and input.is_contiguous()
            and other.is_contiguous()
        ):
            _launch_lastdim_broadcast_kernel(input, other, output)
            return output

        if (
            input.ndim == 3
            and dim == 1
            and input.is_contiguous()
            and other.is_contiguous()
            and input.shape[2] == output.shape[2]
            and other.shape[2] == output.shape[2]
        ):
            if output.shape[2] <= 16:
                if output.is_complex():
                    layout = _get_strided_layout(input, other, output, dim)
                    _launch_linalg_cross_kernel(
                        input,
                        other,
                        output,
                        dim,
                        layout,
                    )
                else:
                    _launch_dim1_small_inner_kernel(input, other, output)
            else:
                _launch_dim1_3d_kernel(input, other, output)
            return output

        layout = _get_strided_layout(input, other, output, dim)
        _launch_linalg_cross_kernel(
            input,
            other,
            output,
            dim,
            layout,
        )
        return output

    input_moved, other_moved, output_dim = _prepare_inputs(input, other, dim)
    output_moved = torch.empty_like(input_moved)
    _launch_linalg_cross_kernel(input_moved, other_moved, output_moved, -1)
    return output_moved.movedim(-1, output_dim).contiguous()


def linalg_cross(input, other, *, dim=-1):
    logger.debug("GEMS_ASCEND LINALG_CROSS")
    return _linalg_cross_impl(input, other, dim)


def linalg_cross_out(input, other, *, dim=-1, out):
    logger.debug("GEMS_ASCEND LINALG_CROSS_OUT")
    if torch._C._is_alias_of(out, input) or torch._C._is_alias_of(out, other):
        raise RuntimeError(
            "linalg_cross: out must not share memory with either input tensor"
        )
    result = _linalg_cross_impl(input, other, dim)
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"linalg_cross: expected out dtype {result.dtype}, but got {out.dtype}"
        )
    if out.device != result.device:
        raise RuntimeError("linalg_cross: out must be on the same device as input")
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out
