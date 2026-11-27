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

import triton

from flag_gems.ops.embedding_bag import _embedding_bag_forward_kernel
from flag_gems.ops.embedding_bag import _embedding_bag_impl as _generic_impl

logger = logging.getLogger(__name__)


def _ignore_device_error(error, count):
    # IX masks unsafe device accesses. No assertion/printf kernel is launched.
    pass


class _MaxForwardLaunch:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # Sequential feature-wise comparison preserves first ties and NaNs
            # without the two cross-row reductions of the generic MAX path.
            kwargs["serial_max"] = True
            return _embedding_bag_forward_kernel[grid](*args, **kwargs)

        return launch


_MAX_FORWARD = _MaxForwardLaunch()


def _embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS_ILUVATAR _EMBEDDING_BAG")
    return _generic_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        # Keep assertion control flow out of the mapping/output kernel on IX.
        _kernel=_MAX_FORWARD if mode == 2 else None,
        _block_config=(
            (
                1,
                min(128, triton.next_power_of_2(max(weight.shape[-1], 1))),
                128,
                2 if weight.shape[-1] <= 64 else 4,
            )
            if mode == 2 and weight.ndim == 2
            else None
        ),
        _async_assert=False,
        _launch_kwargs={"debug": False},
        _check_error=_ignore_device_error,
    )


def _embedding_bag_forward_only(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS_ILUVATAR _EMBEDDING_BAG_FORWARD_ONLY")
    return _generic_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        forward_only=True,
        # Keep assertion control flow out of the mapping/output kernel on IX.
        _kernel=_MAX_FORWARD if mode == 2 else None,
        _block_config=(
            (
                1,
                min(128, triton.next_power_of_2(max(weight.shape[-1], 1))),
                128,
                2 if weight.shape[-1] <= 64 else 4,
            )
            if mode == 2 and weight.ndim == 2
            else None
        ),
        _async_assert=False,
        _launch_kwargs={"debug": False},
        _check_error=_ignore_device_error,
    )
