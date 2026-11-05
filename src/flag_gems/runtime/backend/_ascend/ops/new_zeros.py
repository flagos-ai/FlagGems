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

import torch

from flag_gems.runtime import device as runtime_device

from .zeros import zero_

_NPU_DEVICE_TYPES = {
    runtime_device.name,
    torch._C._get_privateuse1_backend_name(),
}


def new_zeros(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    if not isinstance(self, torch.Tensor):
        raise TypeError("new_zeros expects a Tensor as self")
    if out is None:
        # Let ATen resolve inherited options and validate factory arguments.
        out = torch.ops.aten.new_empty.default(
            self, size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
        )
    elif tuple(out.shape) != tuple(size):
        out.resize_(size)

    if out.device.type not in _NPU_DEVICE_TYPES or out.layout != torch.strided:
        return out.zero_()
    if out.is_complex():
        zero_(out.real)
        zero_(out.imag)
    else:
        zero_(out)
    return out


def new_zeros_out(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    return new_zeros(
        self,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        out=out,
    )
