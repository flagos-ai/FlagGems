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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def _linalg_cross_real_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute a block of contiguous, real three-component cross products."""
    vector_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vector_offsets < num_vectors
    offsets = vector_offsets * 3

    input_0 = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    input_1 = tl.load(input_ptr + offsets + 1, mask=mask, other=0.0)
    input_2 = tl.load(input_ptr + offsets + 2, mask=mask, other=0.0)
    other_0 = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    other_1 = tl.load(other_ptr + offsets + 1, mask=mask, other=0.0)
    other_2 = tl.load(other_ptr + offsets + 2, mask=mask, other=0.0)

    tl.store(output_ptr + offsets, input_1 * other_2 - input_2 * other_1, mask=mask)
    tl.store(output_ptr + offsets + 1, input_2 * other_0 - input_0 * other_2, mask=mask)
    tl.store(output_ptr + offsets + 2, input_0 * other_1 - input_1 * other_0, mask=mask)


@libentry()
@triton.jit
def _linalg_cross_complex_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute a block of cross products over interleaved real/imaginary data."""
    vector_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
    """Use coalesced transactions for interleaved contiguous complex data."""
    vector_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    head_columns = tl.arange(0, 4)
    tail_columns = tl.arange(0, 2)
    vector_mask = vector_offsets < num_vectors
    vector_base = vector_offsets[:, None] * 6
    head_offsets = vector_base + head_columns[None, :]
    tail_offsets = vector_base + 4 + tail_columns[None, :]
    mask = vector_mask[:, None]

    input_head = tl.load(input_ptr + head_offsets, mask=mask, other=0.0)
    input_tail = tl.load(input_ptr + tail_offsets, mask=mask, other=0.0)
    other_head = tl.load(other_ptr + head_offsets, mask=mask, other=0.0)
    other_tail = tl.load(other_ptr + tail_offsets, mask=mask, other=0.0)

    input_head_real, input_head_imag = tl.split(
        tl.reshape(input_head, (BLOCK_SIZE * 2, 2))
    )
    input_0_real, input_1_real = tl.split(
        tl.reshape(input_head_real, (BLOCK_SIZE, 2))
    )
    input_0_imag, input_1_imag = tl.split(
        tl.reshape(input_head_imag, (BLOCK_SIZE, 2))
    )
    input_2_real, input_2_imag = tl.split(input_tail)

    other_head_real, other_head_imag = tl.split(
        tl.reshape(other_head, (BLOCK_SIZE * 2, 2))
    )
    other_0_real, other_1_real = tl.split(
        tl.reshape(other_head_real, (BLOCK_SIZE, 2))
    )
    other_0_imag, other_1_imag = tl.split(
        tl.reshape(other_head_imag, (BLOCK_SIZE, 2))
    )
    other_2_real, other_2_imag = tl.split(other_tail)

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

    output_head = tl.where(
        head_columns[None, :] == 0,
        out_0_real[:, None],
        tl.where(
            head_columns[None, :] == 1,
            out_0_imag[:, None],
            tl.where(
                head_columns[None, :] == 2,
                out_1_real[:, None],
                out_1_imag[:, None],
            ),
        ),
    )
    output_tail = tl.where(
        tail_columns[None, :] == 0, out_2_real[:, None], out_2_imag[:, None]
    )
    tl.store(output_ptr + head_offsets, output_head, mask=mask)
    tl.store(output_ptr + tail_offsets, output_tail, mask=mask)


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
    vector_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
    vector_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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


@triton.jit
def _complex_cross_values(
    input_0_real,
    input_0_imag,
    input_1_real,
    input_1_imag,
    input_2_real,
    input_2_imag,
    other_0_real,
    other_0_imag,
    other_1_real,
    other_1_imag,
    other_2_real,
    other_2_imag,
):
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
    return (
        out_0_real,
        out_0_imag,
        out_1_real,
        out_1_imag,
        out_2_real,
        out_2_imag,
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
    vectors = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vectors < num_vectors
    input_base = (vectors % INPUT_VECTORS) * 3
    other_base = (vectors % OTHER_VECTORS) * 3
    output_base = vectors * 3

    input_0 = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    input_1 = tl.load(input_ptr + input_base + 1, mask=mask, other=0.0)
    input_2 = tl.load(input_ptr + input_base + 2, mask=mask, other=0.0)
    other_0 = tl.load(other_ptr + other_base, mask=mask, other=0.0)
    other_1 = tl.load(other_ptr + other_base + 1, mask=mask, other=0.0)
    other_2 = tl.load(other_ptr + other_base + 2, mask=mask, other=0.0)

    tl.store(output_ptr + output_base, input_1 * other_2 - input_2 * other_1, mask=mask)
    tl.store(
        output_ptr + output_base + 1, input_2 * other_0 - input_0 * other_2, mask=mask
    )
    tl.store(
        output_ptr + output_base + 2, input_0 * other_1 - input_1 * other_0, mask=mask
    )


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
    input_base = (vectors % INPUT_VECTORS) * 6
    other_base = (vectors % OTHER_VECTORS) * 6
    output_base = vectors * 6

    values = _complex_cross_values(
        tl.load(input_ptr + input_base, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + 1, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + 2, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + 3, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + 4, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + 5, mask=mask, other=0.0),
        tl.load(other_ptr + other_base, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + 1, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + 2, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + 3, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + 4, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + 5, mask=mask, other=0.0),
    )
    tl.store(output_ptr + output_base, values[0], mask=mask)
    tl.store(output_ptr + output_base + 1, values[1], mask=mask)
    tl.store(output_ptr + output_base + 2, values[2], mask=mask)
    tl.store(output_ptr + output_base + 3, values[3], mask=mask)
    tl.store(output_ptr + output_base + 4, values[4], mask=mask)
    tl.store(output_ptr + output_base + 5, values[5], mask=mask)


@libentry()
@triton.jit
def _linalg_cross_real_dim1_3d_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    INNER_SIZE: tl.constexpr,
    INPUT_BATCH_STRIDE: tl.constexpr,
    OTHER_BATCH_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vectors = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vectors < num_vectors
    batch = vectors // INNER_SIZE
    inner = vectors % INNER_SIZE
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
def _linalg_cross_complex_dim1_3d_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    INNER_SIZE: tl.constexpr,
    INPUT_BATCH_STRIDE: tl.constexpr,
    OTHER_BATCH_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vectors = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vectors < num_vectors
    batch = vectors // INNER_SIZE
    inner = vectors % INNER_SIZE
    input_base = 2 * (batch * INPUT_BATCH_STRIDE + inner)
    other_base = 2 * (batch * OTHER_BATCH_STRIDE + inner)
    output_base = 2 * (batch * (3 * INNER_SIZE) + inner)
    component_stride = 2 * INNER_SIZE

    values = _complex_cross_values(
        tl.load(input_ptr + input_base, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + 1, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + component_stride, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + component_stride + 1, mask=mask, other=0.0),
        tl.load(input_ptr + input_base + 2 * component_stride, mask=mask, other=0.0),
        tl.load(
            input_ptr + input_base + 2 * component_stride + 1,
            mask=mask,
            other=0.0,
        ),
        tl.load(other_ptr + other_base, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + 1, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + component_stride, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + component_stride + 1, mask=mask, other=0.0),
        tl.load(other_ptr + other_base + 2 * component_stride, mask=mask, other=0.0),
        tl.load(
            other_ptr + other_base + 2 * component_stride + 1,
            mask=mask,
            other=0.0,
        ),
    )
    tl.store(output_ptr + output_base, values[0], mask=mask)
    tl.store(output_ptr + output_base + 1, values[1], mask=mask)
    tl.store(output_ptr + output_base + component_stride, values[2], mask=mask)
    tl.store(output_ptr + output_base + component_stride + 1, values[3], mask=mask)
    tl.store(output_ptr + output_base + 2 * component_stride, values[4], mask=mask)
    tl.store(
        output_ptr + output_base + 2 * component_stride + 1,
        values[5],
        mask=mask,
    )


def _normalize_dim(dim, ndim, name):
    if ndim == 0:
        raise RuntimeError(f"linalg_cross: {name} must have at least one dimension")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected range [{-ndim}, {ndim - 1}], "
            f"but got {dim})"
        )
    return dim % ndim


def _validate_inputs(input, other, dim):
    if input.device != other.device:
        raise RuntimeError("linalg_cross: input and other must be on the same device")
    if input.ndim != other.ndim:
        raise RuntimeError(
            "linalg.cross: inputs must have the same number of dimensions."
        )
    if input.dtype != other.dtype:
        raise RuntimeError("linalg_cross: input and other must have the same dtype")
    if input.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "linalg_cross only supports float32, float64, complex64, and complex128"
        )

    input_dim = _normalize_dim(dim, input.ndim, "input")
    other_dim = _normalize_dim(dim, other.ndim, "other")
    if input.shape[input_dim] != 3 or other.shape[other_dim] != 3:
        raise RuntimeError(
            "linalg_cross: inputs must have length 3 along the cross-product dimension"
        )

    if input.shape == other.shape:
        output_shape = input.shape
    else:
        output_shape = []
        for input_size, other_size in zip(input.shape, other.shape):
            if input_size == other_size or other_size == 1:
                output_shape.append(input_size)
            elif input_size == 1:
                output_shape.append(other_size)
            else:
                raise RuntimeError(
                    "The size of tensor input must match the size of tensor other "
                    "at non-singleton dimensions"
                )
        output_shape = tuple(output_shape)
    return input_dim, output_shape


def _prepare_inputs(input, other, dim):
    """Materialized fallback used only for tensors with more than three dims."""
    input_dim, _ = _validate_inputs(input, other, dim)
    input_moved = input.movedim(input_dim, -1)
    other_moved = other.movedim(input_dim, -1)
    input_moved, other_moved = torch.broadcast_tensors(input_moved, other_moved)
    return input_moved.contiguous(), other_moved.contiguous(), input_dim


def _resolve_view(input):
    if input.is_complex() and input.is_conj():
        input = torch.ops.aten.conj_physical.default.redispatch(
            _FALLBACK_KEYSET, input
        )
    if input.is_neg():
        input = torch.ops.aten.neg.default.redispatch(_FALLBACK_KEYSET, input)
    return input


def _get_strided_layout(input, other, output, dim):
    outer_dims = [axis for axis in range(output.ndim) if axis != dim]
    if len(outer_dims) > 2:
        return None

    input_strides = list(input.stride())
    other_strides = list(other.stride())
    for axis, output_size in enumerate(output.shape):
        if input.shape[axis] == 1 and output_size != 1:
            input_strides[axis] = 0
        if other.shape[axis] == 1 and output_size != 1:
            other_strides[axis] = 0

    if len(outer_dims) == 0:
        outer1_size = 1
        input_outer_strides = (0, 0)
        other_outer_strides = (0, 0)
        output_outer_strides = (0, 0)
    elif len(outer_dims) == 1:
        axis = outer_dims[0]
        outer1_size = output.shape[axis]
        input_outer_strides = (0, input_strides[axis])
        other_outer_strides = (0, other_strides[axis])
        output_outer_strides = (0, output.stride(axis))
    else:
        outer0, outer1 = outer_dims
        outer1_size = output.shape[outer1]
        input_outer_strides = (
            input_strides[outer0],
            input_strides[outer1],
        )
        other_outer_strides = (
            other_strides[outer0],
            other_strides[outer1],
        )
        output_outer_strides = (output.stride(outer0), output.stride(outer1))

    return (
        input_outer_strides,
        other_outer_strides,
        output_outer_strides,
        input.stride(dim),
        other.stride(dim),
        output.stride(dim),
        outer1_size,
    )


def _launch_linalg_cross_kernel(input, other, output, dim, layout=None):
    num_vectors = output.numel() // 3
    if num_vectors == 0:
        return

    block_size = 128 if output.is_complex() and layout is not None else 256
    grid = (triton.cdiv(num_vectors, block_size),)
    with torch_device_fn.device(output.device):
        if output.is_complex():
            input_real = torch.view_as_real(input)
            other_real = torch.view_as_real(other)
            output_real = torch.view_as_real(output)
            if layout is None:
                _linalg_cross_complex_contiguous_kernel[grid](
                    input_real,
                    other_real,
                    output_real,
                    num_vectors,
                    BLOCK_SIZE=block_size,
                    num_warps=8,
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
                    num_warps=4,
                )
        else:
            if layout is None:
                _linalg_cross_real_kernel[grid](
                    input,
                    other,
                    output,
                    num_vectors,
                    BLOCK_SIZE=block_size,
                    num_warps=4,
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
                    num_warps=4,
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
                num_warps=4,
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
                num_warps=4,
            )


def _launch_dim1_3d_kernel(input, other, output):
    num_vectors = output.numel() // 3
    is_small = num_vectors <= 32
    block_size = (
        triton.next_power_of_2(num_vectors)
        if is_small
        else (128 if output.is_complex() else 256)
    )
    num_warps = 1 if is_small else 4
    grid = (triton.cdiv(num_vectors, block_size),)
    input_batch_stride = 0 if input.shape[0] == 1 else input.stride(0)
    other_batch_stride = 0 if other.shape[0] == 1 else other.stride(0)
    with torch_device_fn.device(output.device):
        if output.is_complex():
            _linalg_cross_complex_dim1_3d_kernel[grid](
                torch.view_as_real(input),
                torch.view_as_real(other),
                torch.view_as_real(output),
                num_vectors,
                INNER_SIZE=output.shape[2],
                INPUT_BATCH_STRIDE=input_batch_stride,
                OTHER_BATCH_STRIDE=other_batch_stride,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        else:
            _linalg_cross_real_dim1_3d_kernel[grid](
                input,
                other,
                output,
                num_vectors,
                INNER_SIZE=output.shape[2],
                INPUT_BATCH_STRIDE=input_batch_stride,
                OTHER_BATCH_STRIDE=other_batch_stride,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )


def _linalg_cross_impl(input, other, dim):
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
    """NVIDIA Triton implementation of ``torch.linalg.cross``."""
    logger.debug("GEMS LINALG_CROSS")
    return _linalg_cross_impl(input, other, dim)


def linalg_cross_out(input, other, *, dim=-1, out):
    logger.debug("GEMS LINALG_CROSS_OUT")
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
