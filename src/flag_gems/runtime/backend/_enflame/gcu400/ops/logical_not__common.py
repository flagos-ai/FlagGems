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

from .logical_not import (
    _fake_i64_words,
    _fast_tensor,
    _flat_config,
    _lane_view,
    _launch_flat,
    logical_not_bool_packed_inplace_kernel,
    logical_not_flat_kernel,
    logical_not_func,
    logical_not_i64_word_inplace_kernel,
)

logger = logging.getLogger(__name__)


# In-place variant of the GCU400 logical_not override.  The previous thin
# wrapper dispatched the *generic* 12-CTA block-pointer kernel (its import
# of flag_gems.utils.pointwise_dynamic), which cost 0.07-0.39x vs torch
# native on the mid/large core shapes -- on top of the i1-lane bool store
# tax.  The flat grid-stride kernel writes the 0/1 result back in the
# tensor's own lane dtype (bool through an int8 view, int64 through its
# compact int32 word payload), one kernel launch; the fallback keeps the
# *backend* 48-CTA pointwise path (never the generic 12-CTA one).
def logical_not_(A):
    logger.debug("GEMS_ENFLAME LOGICAL_NOT_")
    if _fast_tensor(A):
        n_total = A.numel()
        if n_total == 0:
            return A
        if A.dtype == torch.int64:
            if _fake_i64_words(A):
                config = _flat_config(n_total, 4)
                if config is not None:
                    block, grid = config
                    _launch_flat(
                        logical_not_i64_word_inplace_kernel,
                        grid,
                        (A, n_total),
                        block,
                    )
                    return A
            return logical_not_func(A, out0=A)
        if A.dtype == torch.bool and n_total % 4 == 0 and A.data_ptr() % 4 == 0:
            n_words = n_total // 4
            config = _flat_config(n_words, 4)
            if config is not None:
                block, grid = config
                _launch_flat(
                    logical_not_bool_packed_inplace_kernel,
                    grid,
                    (A, n_words),
                    block,
                )
                return A
        config = _flat_config(n_total, A.element_size())
        if config is not None:
            block, grid = config
            lanes = _lane_view(A)
            _launch_flat(
                logical_not_flat_kernel,
                grid,
                (lanes, lanes, n_total),
                block,
            )
            return A
    return logical_not_func(A, out0=A)
