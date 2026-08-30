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


# ── scatter kernel: zero-init output, then write only the "1" positions ───────


@libentry()
@triton.jit
def one_hot_scatter_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter one-hot kernel.

    For each input element computes ``row * num_classes + index`` and stores a
    single ``1`` at that position.  The output buffer must be zero-initialized
    before this kernel is launched.

    This approach is always used regardless of ``num_classes`` because the
    tiny per-element overhead of the scatter write (1 extra store per row)
    is vastly outweighed by eliminating the broadcast-comparison cost of the
    dense alternative on MTHREADS hardware.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    safe_indices = tl.maximum(tl.minimum(indices, num_classes - 1), 0)
    out_offsets = offsets * num_classes + safe_indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


# ── block-size helper ──────────────────────────────────────────────────────────


def _scatter_block_size(num_elements: int) -> int:
    """Pick an optimal BLOCK_SIZE for the scatter kernel.

    On small inputs more grid blocks exploit multi-processor parallelism
    (192 KB shared memory per SM removes register-pressure concerns).
    On large inputs a bigger block amortises launch overhead.
    """
    if num_elements <= 4096:
        return 256
    elif num_elements <= 32768:
        return 512
    elif num_elements <= 262144:
        return 1024
    else:
        return 2048


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

    # Infer / validate num_classes (single device → host sync).
    if num_classes == -1:
        # Use aminmax when available (one kernel for both min & max, PyTorch ≥ 1.11).
        try:
            amin, amax = tensor.aminmax()
            if int(amin.item()) < 0:
                raise RuntimeError("Class values must be non-negative.")
            num_classes = int(amax.item()) + 1
        except AttributeError:
            maxv = int(tensor.max().item())
            num_classes = maxv + 1
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
    else:
        if (tensor >= num_classes).any():
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
        # Scatter approach: zero-init then write only the "1" positions.
        # This avoids the expensive broadcast comparison in dense kernels
        # and is consistently faster on MTHREADS hardware.
        out = torch.zeros(
            num_elements * num_classes,
            device=tensor.device,
            dtype=torch.int64,
        )
        BLOCK_SIZE = _scatter_block_size(num_elements)
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
        one_hot_scatter_kernel[grid](
            flat_input,
            out,
            num_elements,
            num_classes,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out.view(*tensor.shape, num_classes)
