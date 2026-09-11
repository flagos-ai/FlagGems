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

logger = logging.getLogger(__name__)


def select_backward(grad, input_sizes, dim, index, out=None):
    logger.debug("GEMS_ASCEND SELECT_BACKWARD")
    dim = int(dim)
    index = int(index)
    sizes = list(input_sizes)
    ndim = len(sizes)

    assert dim >= -ndim and dim < ndim, "Invalid dim"
    dim %= ndim

    dim_size = sizes[dim]

    assert index >= -dim_size and index < dim_size, "Invalid index"
    index %= dim_size

    if out is None:
        out = torch.empty(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        assert tuple(out.shape) == tuple(sizes), "out shape mismatch"
        assert out.dtype == grad.dtype, "dtype mismatch"
        assert out.device == grad.device, "device mismatch"

    out.zero_()
    out.select(dim, index).copy_(grad)
    return out
