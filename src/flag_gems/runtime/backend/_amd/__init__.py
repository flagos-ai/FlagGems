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
    vendor_name="amd",
    device_name="cuda",
    device_query_cmd="rocm-smi",
)

"""
Mapping from an AMD GPU architecture to the directory holding that
architecture's specialized configuration.

Keys are looked up from the most to the least specific: the exact target torch
reports in gcnArchName, then "major.minor", then the major version alone. An
architecture absent from the map falls back to the vendor-wide configuration.

Targets are listed individually because the major version alone is too coarse:
gfx1250 and gfx1251 also report major 12, yet they form the separate GFX12.5
family (LLVM keeps them out of gfx12-generic) with a much larger addressable
LDS, so the RDNA4 candidates would be a poor fit for them.
"""

ARCH_MAP = {
    "gfx1200": "rdna4",
    "gfx1201": "rdna4",
}

CUSTOMIZED_UNUSED_OPS = (
    "add",
    "cos",
    "cumsum",
)


__all__ = ["*"]
