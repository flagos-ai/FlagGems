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

import torch  # noqa: F401
import torch_mlu  # noqa: F401

from flag_gems.runtime.backend.backend_utils import VendorDescriptor  # noqa: E402

from .utils import DEVICE_COUNT  # noqa: F401
from .utils import MAX_GRID_SIZE_X  # noqa: F401
from .utils import MAX_GRID_SIZE_Y  # noqa: F401
from .utils import MAX_GRID_SIZE_Z  # noqa: F401
from .utils import MAX_GRID_SIZES  # noqa: F401
from .utils import MAX_NRAM_SIZE  # noqa: F401
from .utils import TOTAL_CLUSTER_NUM  # noqa: F401
from .utils import TOTAL_CORE_NUM  # noqa: F401

try:
    from torch_mlu.utils.model_transfer import transfer  # noqa: F401
except ImportError:
    pass

vendor_info = VendorDescriptor(
    vendor_name="cambricon",
    device_name="mlu",
    device_query_cmd="cnmon",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "masked_scatter",
    "masked_scatter_",
    "scatter_add_",
)

__all__ = ["*"]
