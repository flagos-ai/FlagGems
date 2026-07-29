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

import importlib.util
import os
import re

from backend_utils import VendorDescriptor

# NOTE: transfer_to_gcu is not used anywhere
# try:
#     from torch_gcu import transfer_to_gcu  # noqa: F401
# except Exception:
#    logger.warning("torch_gcu not installed")

# TODO: Revise the following imports to be exception free
if importlib.util.find_spec("triton.backends.enflame") is None:
    from triton_gcu.triton.driver import _GCUDriver
else:
    from triton.backends.enflame.driver import _GCUDriver

driver = _GCUDriver()
arch = driver.get_arch()
arch_version = int(re.search(r"gcu(\d+)", arch).group(1))

vendor_info = VendorDescriptor(
    vendor_name="enflame",
    device_name="gcu",
    device_query_cmd="",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    int64_enabled=False,
    tle_enabled=True,
)

os.environ["ARCH"] = str(arch_version)
ARCH_MAP = {"3": "gcu300", "4": "gcu400"}
# i64 to/copy is not supported in gcu300
if arch_version == 300:
    CUSTOMIZED_UNUSED_OPS = (
        "_to_copy",
        "to_copy",
        "copy_",
    )
elif arch_version == 400 or arch_version == 410:
    CUSTOMIZED_UNUSED_OPS = (
        "to_copy",
        "copy_",
    )

__all__ = ["*"]
