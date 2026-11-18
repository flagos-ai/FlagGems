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
#

"""Out variant sharing the functional Dirichlet numerical primitive."""

import logging

import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

from .dirichlet_grad import _dirichlet_grad_one, _validate_inputs

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def _dirichlet_grad_func(x, alpha, total):
    return _dirichlet_grad_one(x, alpha, total).to(x.dtype)


def _dirichlet_grad_into(x, alpha, total, *, out):
    logger.debug("GEMS_KUNLUNXIN DIRICHLET_GRAD_OUT")
    shape = _validate_inputs(x, alpha, total)
    if out.device != x.device or out.dtype != x.dtype:
        raise RuntimeError("_dirichlet_grad: out must match input device and dtype")
    if tuple(out.shape) != tuple(shape):
        out.resize_(shape)
    with torch_device_fn.device(x.device):
        _dirichlet_grad_func(x, alpha, total, out0=out)
    return out
