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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Fused validation allocates the output before reporting an invalid index.
# Keep that speculative allocation at or below 64 MiB.  S5000 crossover tests
# remain favorable through 16M elements and first approach parity near 32M.
_FUSED_VALIDATION_MAX_OUTPUT_ELEMENTS = 8_388_608


# ── dense comparison kernels (specialised for 16/32/64 classes) ──────────────


@libentry()
@triton.jit
def one_hot_kernel_16(
    input_ptr,
    output_ptr,
    num_elements,
    actual_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_base = offsets * actual_classes

    class_offsets = tl.arange(0, 16)
    out_offsets = out_base[:, None] + class_offsets[None, :]
    values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
    valid_classes = class_offsets < actual_classes
    combined_mask = mask[:, None] & valid_classes[None, :]
    tl.store(output_ptr + out_offsets, values, mask=combined_mask)


@libentry()
@triton.jit
def one_hot_kernel_32(
    input_ptr,
    output_ptr,
    num_elements,
    actual_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_base = offsets * actual_classes

    class_offsets = tl.arange(0, 32)
    out_offsets = out_base[:, None] + class_offsets[None, :]
    values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
    valid_classes = class_offsets < actual_classes
    combined_mask = mask[:, None] & valid_classes[None, :]
    tl.store(output_ptr + out_offsets, values, mask=combined_mask)


@libentry()
@triton.jit
def one_hot_kernel_64(
    input_ptr,
    output_ptr,
    num_elements,
    actual_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_base = offsets * actual_classes

    class_offsets = tl.arange(0, 64)
    out_offsets = out_base[:, None] + class_offsets[None, :]
    values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
    valid_classes = class_offsets < actual_classes
    combined_mask = mask[:, None] & valid_classes[None, :]
    tl.store(output_ptr + out_offsets, values, mask=combined_mask)


# ── scatter kernel: only write the "1" positions (output must be zeroed first) ──


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
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_offsets = offsets * num_classes + indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


@libentry()
@triton.jit
def one_hot_validate_set_one_kernel(
    input_ptr,
    output_ptr,
    status_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """Validate indices while scattering into a zero-initialized output.

    Status values preserve the existing error precedence: negative indices
    take priority over indices greater than or equal to ``num_classes``.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    negative = indices < 0
    too_large = indices >= num_classes
    valid = mask & ~negative & ~too_large
    safe_indices = tl.minimum(tl.maximum(indices, 0), num_classes - 1)
    tl.store(
        output_ptr + offsets * num_classes + safe_indices,
        1,
        mask=valid,
    )
    code = tl.where(negative, 2, tl.where(too_large, 1, 0))
    block_code = tl.max(tl.where(mask, code, 0), axis=0)
    tl.atomic_max(status_ptr, block_code)


# ── block-size helpers ─────────────────────────────────────────────────────────


def _dense_block_size(num_elements: int, num_classes: int) -> int:
    """Pick a BLOCK_SIZE for dense kernels.

    The kernel writes ``BLOCK_SIZE * num_classes`` values per block.  We cap
    the per-block output to ~32 kB so that it stays L1-resident.
    """
    if num_elements <= 512:
        base = 64
    elif num_elements <= 4096:
        base = 128
    elif num_elements <= 32768:
        base = 256
    else:
        base = 512
    max_rows = max(32, (32 * 1024) // (num_classes * 8))
    max_rows = 1 << (max_rows.bit_length() - 1)
    return min(base, max_rows)


def _scatter_block_size(num_elements: int) -> int:
    """Pick a BLOCK_SIZE for the scatter-only kernel."""
    if num_elements <= 1024:
        return 256
    elif num_elements <= 16384:
        return 512
    else:
        return 1024


# ── main entry point ───────────────────────────────────────────────────────────


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
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

    use_fused_validation = False
    inferred_num_classes = num_classes == -1
    if inferred_num_classes:
        if tensor.device.type == "cpu":
            num_classes = int(tensor.max().item()) + 1
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
        else:
            minv, maxv = tensor.aminmax()
            minv, maxv = int(minv.item()), int(maxv.item())
            if minv < 0:
                raise RuntimeError("Class values must be non-negative.")
            num_classes = maxv + 1
    elif (
        num_classes > 0
        and tensor.device.type != "cpu"
        and tensor.numel() * num_classes <= _FUSED_VALIDATION_MAX_OUTPUT_ELEMENTS
    ):
        use_fused_validation = True
    else:
        invalid = (tensor < 0) | (tensor >= num_classes)
        if invalid.any():
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
            else:
                raise RuntimeError("Class values must be smaller than num_classes.")

    if num_classes < 1:
        raise RuntimeError("num_classes should be positive")

    if tensor.device.type == "cpu":
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    flat_input = tensor.contiguous().view(-1)
    num_elements = flat_input.numel()

    with torch_device_fn.device(tensor.device):
        if use_fused_validation:
            out = torch.zeros(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            status = torch.zeros(1, device=tensor.device, dtype=torch.int32)
            BLOCK_SIZE = _scatter_block_size(num_elements)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_validate_set_one_kernel[grid](
                flat_input,
                out,
                status,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            error = int(status.item())
            if error == 2:
                raise RuntimeError("Class values must be non-negative.")
            if error == 1:
                raise RuntimeError("Class values must be smaller than num_classes.")
        elif inferred_num_classes:
            out = torch.zeros(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = _scatter_block_size(num_elements)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_set_one_kernel[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif num_classes <= 16:
            out = torch.empty(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = _dense_block_size(num_elements, num_classes)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_kernel_16[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif num_classes <= 32:
            out = torch.empty(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = _dense_block_size(num_elements, num_classes)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_kernel_32[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif num_classes <= 64:
            out = torch.empty(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = _dense_block_size(num_elements, num_classes)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_kernel_64[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            out = torch.zeros(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = _scatter_block_size(num_elements)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_set_one_kernel[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out.view(*tensor.shape, num_classes)
