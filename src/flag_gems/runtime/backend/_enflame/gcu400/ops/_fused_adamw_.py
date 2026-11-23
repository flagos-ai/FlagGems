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

"""Enflame gcu400 fused AdamW.

The generic Triton implementation already runs correctly on GCU; this module
re-exports it so the arch module and ``flag_gems.ops._fused_adamw_`` stay the
same function objects. Shadowing the generic module with a byte-identical
copy broke ATen-dispatch tracing and monkeypatching against the generic
module (tests/test_fused_adamw_.py::test_fused_adamw_dispatch), which is the
single implementation every other backend registers. Vendor-specific tuning,
when needed, should reintroduce a local kernel here.
"""

from flag_gems.ops._fused_adamw_ import _fused_adamw_, _fused_adamw__tensor_lr

__all__ = ["_fused_adamw_", "_fused_adamw__tensor_lr"]
