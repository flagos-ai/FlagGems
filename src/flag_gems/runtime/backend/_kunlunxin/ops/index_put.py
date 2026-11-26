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

from flag_gems.ops.index_put import index_put as _common_index_put
from flag_gems.ops.index_put import index_put_ as _common_index_put_

from .index_put_impl import _index_put_impl_

logger = logging.getLogger(__name__)


def index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS_KUNLUNXIN INDEX_PUT")
    indices = list(indices)
    # `_index_put_impl_` fast paths assume all-tensor indices (no None markers);
    # route the general cases (empty / None markers) to the common implementation.
    if not indices or any(index is None for index in indices):
        return _common_index_put(inp, indices, values, accumulate)
    out = inp.clone()
    return _index_put_impl_(out, indices, values, accumulate)


def index_put_(inp, indices, values, accumulate=False):
    logger.debug("GEMS_KUNLUNXIN INDEX_PUT_")
    indices = list(indices)
    if not indices or any(index is None for index in indices):
        return _common_index_put_(inp, indices, values, accumulate)
    return _index_put_impl_(inp, indices, values, accumulate)
