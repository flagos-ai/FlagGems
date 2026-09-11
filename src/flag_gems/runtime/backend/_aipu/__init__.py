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

from backend_utils import VendorDescriptor  # noqa: E402
from triton.runtime import driver  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="aipu",
    device_name="aipu",
    device_query_cmd="aipu",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    bf16_enabled=False,
    int64_enabled=False,
)

# The aipu backend is loaded dynamically, so here need to active first.
driver.active.get_active_torch_device()

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
