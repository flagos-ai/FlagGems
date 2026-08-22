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

import torch
import triton.language as tl

from flag_gems.utils.triton_version_utils import has_triton_tle


def get_triton_dtype(dtype):
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    return dtype_map.get(dtype, None)


def tle_interfaces_available() -> bool:
    if not has_triton_tle(3, 6, 0):
        return False

    try:
        import triton.experimental.tle.language as _tle  # noqa: F401
    except ImportError:
        return False

    required = [
        ("gpu", _tle),
        ("pipe", _tle),
        ("wgmma", _tle.gpu),
        ("wgmma_wait", _tle.gpu),
        ("copy", _tle.gpu),
        ("alloc", _tle.gpu),
        ("smem", _tle.gpu),
        ("warp_specialize", _tle.gpu),
    ]
    return all(hasattr(obj, name) for name, obj in required)
