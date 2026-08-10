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

from .repeat_impl import repeat as _original_repeat

# the uplimit f32 can present the precision of i32
_F32_PRECISION_NUMEL_THRESHOLD = 2**24


def _compute_output_numel(x, sizes):
    in0_shape = list(x.shape)
    sizes_shape = list(sizes)
    if len(sizes_shape) > len(in0_shape):
        diff = len(sizes_shape) - len(in0_shape)
        in0_shape = [1] * diff + in0_shape
    out_shape = [s * d for s, d in zip(in0_shape, sizes_shape)]
    return math.prod(out_shape)


def repeat(inp, sizes):
    original_precision_priority = os.environ.get("PRECISION_MODE", None)

    out_numel = _compute_output_numel(inp, sizes)
    if out_numel > _F32_PRECISION_NUMEL_THRESHOLD:
        os.environ["PRECISION_MODE"] = "1"

    try:
        return _original_repeat(inp, sizes)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_MODE"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_MODE", None)
