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

import math
import os

from .tile_impl import tile as _original_tile

# the uplimit f32 can present the precision of i32
_F32_PRECISION_NUMEL_THRESHOLD = 2**24


def _compute_output_numel(x, dims):
    in0_shape = list(x.shape)
    dims_shape = list(dims)
    diff = len(in0_shape) - len(dims_shape)
    if diff > 0:
        dims_shape = [1] * diff + dims_shape
    elif diff < 0:
        in0_shape = [1] * (-diff) + in0_shape
    out_shape = [s * d for s, d in zip(in0_shape, dims_shape)]
    return math.prod(out_shape)


def tile(inp, dims):
    original_precision_priority = os.environ.get("PRECISION_MODE", None)
    out_numel = _compute_output_numel(inp, dims)
    if out_numel > _F32_PRECISION_NUMEL_THRESHOLD:
        os.environ["PRECISION_MODE"] = "1"

    try:
        return _original_tile(inp, dims)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_MODE"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_MODE", None)
