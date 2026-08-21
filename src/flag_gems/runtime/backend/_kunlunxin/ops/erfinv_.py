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

"""Kunlunxin erfinv_ (aten::erfinv_) vendor override.

The in-place variant shares the exact same kernel entry as erfinv (see
`erfinv.py`): the previous libdevice-based kernel here (~0.4x dtype-equal)
has been superseded by the Chebyshev-24/Clenshaw (fp32) and shifted-power-
basis Horner (fp16/bf16) polynomial kernel introduced with erfinv.  This
module only keeps the `erfinv_` name bound for `ops/__init__.py`.
"""

from .erfinv import erfinv_  # noqa: F401
