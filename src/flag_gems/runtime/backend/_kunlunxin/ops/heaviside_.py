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


@triton.jit
def _heaviside_inplace_kernel(x_ptr, v_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    # heaviside(x, v) = 0 if x < 0; v if x == 0 (incl. -0.0, which compares
    # equal to 0); 1 if x > 0.  For NaN inputs no comparison holds, so the
    # result is 0, matching ATen's CPU/CUDA kernels.  (The generic kernel's
    # `x + x` else-branch propagates NaN instead.)
    # Explicit .to(dtype) + single-level where measures 5-18% faster than
    # int-scalar/nested-where forms on XPU for fp16/bf16 (min-of-3 A/B).
    step = (x > 0).to(x.dtype)
    res = tl.where(x == 0, v, step)
    tl.store(x_ptr + offsets, res, mask=mask)


def _expand_values(values: torch.Tensor, self: torch.Tensor) -> torch.Tensor:
    """Broadcast ``values`` to ``self``'s shape as a contiguous device tensor.

    NOTE: ``Tensor.contiguous()`` on a broadcast (stride-0) view funnels into
    ``aten::_to_copy`` / ``copy_`` / ``cat``, which are intercepted by
    flag_gems (vendor ``to``/``copy``/``cat``) and crash on XPU for
    fp16/bf16 expanded copies of (1,N)/(N,1)/() -> (512,512)-class shapes
    (illegal memory access, kernel exception).  ``aten::_copy_from`` is the
    native strided-copy primitive that flag_gems never overrides, so it is
    used explicitly to materialize the expanded values.
    """
    v_exp = values.expand_as(self)
    if v_exp.is_contiguous():
        return v_exp
    dst = torch.empty_like(self)
    torch.ops.aten._copy_from(v_exp, dst, False)
    return dst


def heaviside_(self: torch.Tensor, values: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN HEAVISIDE_")

    if not torch.is_tensor(values):
        values = torch.as_tensor(values, device=self.device, dtype=self.dtype)
    elif values.device != self.device or values.dtype != self.dtype:
        values = values.to(device=self.device, dtype=self.dtype)

    n_elements = self.numel()
    if n_elements == 0:
        return self

    v_tensor = _expand_values(values, self)

    # Materialize a contiguous copy only when self is not contiguous.  The
    # write-back path also uses _copy_from (not self.copy_, which would be
    # routed through the flag_gems copy_ override).
    if self.is_contiguous():
        x_contig = self
    else:
        x_contig = torch.empty_like(self)
        torch.ops.aten._copy_from(self, x_contig, False)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(self.device):
        _heaviside_inplace_kernel[grid](
            x_contig.view(-1), v_tensor.view(-1), n_elements, BLOCK_SIZE=1024
        )

    if x_contig is not self:
        torch.ops.aten._copy_from(x_contig, self, False)

    return self
