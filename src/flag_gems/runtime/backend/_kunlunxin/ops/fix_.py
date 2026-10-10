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

from flag_gems.ops.copy import copy_ as _triton_copy_
from flag_gems.ops.fix_ import _fix_inplace_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems.ops.fix_")


def _fix_real_(t: torch.Tensor):
    """Run the in-place trunc kernel over a real floating tensor.

    Uses a triton copy (not ``aten::copy_``) to write results back when ``t``
    is not contiguous.
    """
    n_elements = t.numel()
    if n_elements == 0:
        return

    t_contig = t.contiguous()

    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(t.device):
        _fix_inplace_kernel[grid](t_contig, n_elements, BLOCK_SIZE=1024)

    if t_contig is not t:
        _triton_copy_(t, t_contig)


def fix_(self: torch.Tensor):
    """
    Wrapper for ATen operator: ('fix_', <Autograd.disable: False>)
    In-place truncation toward zero for floating tensors.
    Integer tensors are left unchanged (no-op).
    """
    logger.debug("GEMS_KUNLUNXIN FIX_")

    if self.is_complex():
        _fix_real_(torch.view_as_real(self))
        return self

    if not self.is_floating_point():
        return self

    _fix_real_(self)
    return self
