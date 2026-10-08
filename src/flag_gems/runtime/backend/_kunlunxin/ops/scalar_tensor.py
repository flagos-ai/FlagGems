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

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["value"])
def scalar_tensor_kernel(out_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr):
    """Vector-shaped single-element store.

    The store is expressed as a [BLOCK_SIZE]-wide masked store rather than a
    scalar ``tl.store(out_ptr, value)``.  On this backend the scalar form is
    ~3x more expensive to launch (measured 13.2us vs 4.4us for a 0-dim
    tensor); the store shape, not the launch config, is what costs.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, value, mask=mask)


def scalar_tensor(s, *, dtype=None, layout=None, device=None, pin_memory=None):
    """0-dim (scalar) tensor creation, Kunlunxin(XPU) optimized.

    The generic implementation allocates through the FlagGems ``aten::empty``
    override and then runs a ``pointwise_dynamic`` Triton kernel to store the
    scalar value; on XPU those two launches cost ~20us vs ~5us for native
    ``torch.scalar_tensor``.

    This XPU specialization:
      1. allocates natively with ``torch.empty_strided`` (``empty`` /
         ``empty_strided`` are not overridden by FlagGems), avoiding the
         empty-kernel launch entirely;
      2. stores the scalar with a dedicated minimal Triton kernel launched in
         the same shape as the vendor ``full_kernel`` (12-program grid,
         BLOCK_SIZE=1).  ``out.fill_(s)`` cannot be used here: FlagGems
         registers ``fill_.Scalar`` at the CUDA dispatch key, and
         ``torch._C._AutoDispatchBelowAutograd`` only excludes the *Autograd*
         key, so an in-place ``fill_`` still lands in FlagGems' generic
         fill kernel (~+8-12us).

    Both steps run on the XPU device; no CPU/ATen/native/composite fallback.
    """
    logger.debug("GEMS_KUNLUNXIN SCALAR_TENSOR")
    out = torch.empty_strided(
        (), (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    if dtype == torch.bool:
        s = bool(s)
    elif dtype is None or dtype.is_floating_point:
        try:
            s = float(s)
        except (TypeError, ValueError, OverflowError):
            pass
    elif not isinstance(s, int):
        # numpy integer / 0-dim tensor: Triton needs a plain Python scalar.
        s = s.item() if hasattr(s, "item") else s
    with torch_device_fn.device(out.device):
        scalar_tensor_kernel[(12, 1, 1)](
            out,
            1,
            s,
            BLOCK_SIZE=1,
            buffer_size_limit=2048,
            isCloseDtypeConvert=True,
        )
    return out
