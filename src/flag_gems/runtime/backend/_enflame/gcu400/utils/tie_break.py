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

"""NaN-propagating min/max-with-index combines for GCU scan kernels.

The shared ``flag_gems.utils.triton_lang_extension`` versions build their mask
with ``mask |= a_isnan and not b_isnan``.  That combine is correct when called
as a plain binary op, but inside ``tl.associative_scan`` on triton-gcu 3.6 the
NaN terms are silently dropped and the scan degrades to a plain min/max
(measured: ``[1, nan, 0, 2, nan, -1, 3, nan]`` scanned to ``[1, 1, 0, 0, 0,
-1, -1, -1]`` instead of ``torch.cummin``'s ``[1, nan, nan, nan, nan, nan, nan,
nan]``).

The working shape is: substitute NaN with +-inf so the ordering is a single
comparison over the substituted values, and add one term so a real +-inf never
displaces a NaN.  Verified against ``torch.cummin``/``torch.cummax`` including
real +-inf inputs and the "later index wins a tie" rule.
"""

import triton
import triton.language as tl


@triton.jit
def minimum_with_index_tie_break_right(a_value, a_index, b_value, b_index):
    if a_value.dtype.is_floating():
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        av = tl.where(a_isnan, float("-inf"), a_value)
        bv = tl.where(b_isnan, float("-inf"), b_value)
        later = (a_index > b_index) | (a_isnan & (~b_isnan))
    else:
        av = a_value
        bv = b_value
        later = a_index > b_index
    mask = (av < bv) | ((av == bv) & later)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index_tie_break_right(a_value, a_index, b_value, b_index):
    if a_value.dtype.is_floating():
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        av = tl.where(a_isnan, float("inf"), a_value)
        bv = tl.where(b_isnan, float("inf"), b_value)
        later = (a_index > b_index) | (a_isnan & (~b_isnan))
    else:
        av = a_value
        bv = b_value
        later = a_index > b_index
    mask = (av > bv) | ((av == bv) & later)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)
