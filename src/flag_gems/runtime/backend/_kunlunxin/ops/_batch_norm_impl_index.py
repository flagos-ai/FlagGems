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
    # aten::_batch_norm_impl_index updates the running statistics in *every*
    # input dtype (not just float32) and uses the unbiased variance estimate
    # (n / (n - 1)) for `running_var`; `save_var` keeps the biased
    # `1/sqrt(var + eps)`. The local `batch_norm` wrapper defaults both
    # behaviours off, so opt in explicitly here.
    output, save_mean, save_var = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        update_running_all_dtypes=True,
        unbiased_running_var=True,
    )
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output, save_mean, save_var, reserve, 0


import sys as _sys  # noqa: E402

# The platform benchmarks (benchmark/test_batch_norm_impl_index.py) bind their
# candidate from the *generic* module path
# (`from flag_gems.ops._batch_norm_impl_index import batch_norm_impl_index`)
# instead of `flag_gems._batch_norm_impl_index`. On XPU3 the generic
# `batch_norm_forward_kernel` does not lower at all (`TritonXPUUnrollControl` /
# `uni_sram` pass failure), so that path can never be measured or executed
# here even though the registrar already routes `flag_gems._batch_norm_impl_index`
# to this module. Rebind the generic module attribute to this vendor
# implementation so both call paths resolve to the kernel that actually runs on
# XPU (same pattern as `_enflame/gcu300/ops/div.py`, which rebinds
# `flag_gems.ops.div_mode`).
_generic_module = _sys.modules.get("flag_gems.ops._batch_norm_impl_index")
if _generic_module is not None:
    _generic_module.batch_norm_impl_index = batch_norm_impl_index
