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

from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="nvidia",
    device_name="cuda",
    device_query_cmd="",
    tle_enabled=True,
)

"""
Mapping from NVIDIA GPU compute capability major version
to architecture codename.

Example:
  8.x -> Ampere (A100)
  9.x -> Hopper (H100)
"""

ARCH_MAP = {
    "9": "hopper",
    "8": "ampere",
}


"""
Tuple of operation names to exclude,  empty tuple means all operations are enabled.

Example:
    CUSTOMIZED_UNUSED_OPS = ("add", "cos")
"""

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
