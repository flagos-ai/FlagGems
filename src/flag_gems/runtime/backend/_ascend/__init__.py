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

from backend_utils import VendorDescriptor  # noqa: F402

from .utils import CORE_NUM  # noqa: F401


def get_triton_extra_name():
    try:
        import triton
        from packaging import version

        if version.parse(triton.__version__) < version.parse("3.2.0"):
            return "ascend"
        else:
            return "cann"
    except Exception:
        return "ascend"


vendor_info = VendorDescriptor(
    vendor_name="ascend",
    device_name="npu",
    device_query_cmd="npu-smi info",
    dispatch_key="PrivateUse1",
    triton_extra_name=get_triton_extra_name(),
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "to_copy",
    "contiguous",
    "copy_",
    "_to_copy",
    "sort",
    "sort_stable",
    "topk",
)


__all__ = ["*"]
