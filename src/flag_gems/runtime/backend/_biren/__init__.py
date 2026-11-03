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

import torch_supa
from backend_utils import VendorDescriptor

torch_supa._C._transfer.device_type(True)

vendor_info = VendorDescriptor(
    vendor_name="biren",
    device_name="cuda",
    device_query_cmd="brsmi",
    dispatch_key="PrivateUse1",
    triton_extra_name="supa",  # tl.extra.supa...
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ("copy_",)

__all__ = ["*"]
