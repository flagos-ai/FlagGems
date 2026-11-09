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

from flag_gems.ops._aminmax import _aminmax as _generic_aminmax
from flag_gems.ops._aminmax import _aminmax_out as _generic_aminmax_out

logger = logging.getLogger(__name__)

__all__ = ["_aminmax", "_aminmax_out"]


def _aminmax(inp):
    """gcu400 ``_aminmax``.

    GCU has no fp64 support: torch_gcu already runs fp64 aten ops with a float
    replacement, but the generated Triton kernel's fp64 NaN-detection
    (``min_val != min_val``) fails to legalize.  Reduce an fp32 view (the same
    values the GCU reference sees) and restore the input dtype.
    """
    if inp.dtype != torch.float64:
        return _generic_aminmax(inp)
    min_out, max_out = _generic_aminmax(inp.to(torch.float32))
    return min_out.to(torch.float64), max_out.to(torch.float64)


def _aminmax_out(inp, *, out0, out1):
    if inp.dtype != torch.float64:
        return _generic_aminmax_out(inp, out0=out0, out1=out1)
    min_out, max_out = _aminmax(inp)
    out0.resize_(())
    out1.resize_(())
    out0.copy_(min_out)
    out1.copy_(max_out)
    return out0, out1
