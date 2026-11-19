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

"""Explicit IFWA selection for the FlagGems library."""

import torch
import torch_ifwa  # noqa: F401  # Registers torch.ifwa and PrivateUse1.
from backend_utils import VendorDescriptor

if not hasattr(torch, "ifwa") or not torch.ifwa.is_available():
    raise RuntimeError("IFWA runtime is unavailable; no alternate backend is permitted")

vendor_info = VendorDescriptor(
    vendor_name="intellifusion",
    device_name="ifwa",
    device_query_cmd="false",  # Select explicitly with GEMS_VENDOR=intellifusion.
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    fp8_enabled=False,
    complex_enabled=False,
)
CUSTOMIZED_UNUSED_OPS = ()
