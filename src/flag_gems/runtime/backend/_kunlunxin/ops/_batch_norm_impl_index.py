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

from .batch_norm import batch_norm

logger = logging.getLogger(__name__)


def batch_norm_impl_index(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    cudnn_enabled=True,
):
    """Kunlunx/XPU override of ``aten::_batch_norm_impl_index``.

    The generic ``batch_norm_forward_kernel`` (2D [BLOCK_M, BLOCK_N] tile, grid =
    feat_dim) does not lower on the XPU compiler for most shapes (pass failures
    in ``TritonXPULegalize`` / uni_sram OOM), which made ``F.batch_norm`` /
    ``torch.batch_norm`` fail at runtime. This override delegates to the vendor
    ``batch_norm`` wrapper: inference (training=False) is a single per-(n, c)
    slice kernel launch (contiguous block DMA), training keeps the vendor 3-stage
    stats/combine/normalize path. Return tuple matches the generic contract:
    (output, save_mean, save_var, reserve, impl_index).
    """
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_IMPL_INDEX")
    output, save_mean, save_var = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output, save_mean, save_var, reserve, 0