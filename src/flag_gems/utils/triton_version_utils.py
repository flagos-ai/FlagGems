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

import re

import triton
from packaging.version import InvalidVersion, Version

from flag_gems.runtime import backend
from flag_gems.runtime.backend.device_finder import DeviceDetector


def _coerce_triton_version(version: str) -> Version:
    try:
        return Version(version)
    except InvalidVersion:
        release = []
        for part in version.split("+", 1)[0].split(".")[:3]:
            match = re.match(r"\d+", part)
            release.append(match.group(0) if match else "0")
        while len(release) < 3:
            release.append("0")
        return Version(".".join(release))


def _triton_version_at_least(major: int, minor: int, patch: int = 0) -> bool:
    version = str(getattr(triton, "__version__", "0.0.0"))
    return _coerce_triton_version(version) >= Version(f"{major}.{minor}.{patch}")


def is_support_vendor():
    device = DeviceDetector()
    vendor_info = backend.get_vendor_info(device.vendor_name)
    return vendor_info.tle_enabled


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    if not _triton_version_at_least(major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language as _tle  # noqa: F401

        return is_support_vendor()
    except ImportError:
        return False


HAS_TLE = has_triton_tle()


def has_tle_device_mesh() -> bool:
    """Check if TLE device_mesh is available."""
    if not HAS_TLE:
        return False
    try:
        import triton.experimental.tle.language as tle_exp

        return hasattr(tle_exp, "device_mesh")
    except ImportError:
        return False


HAS_TLE_DEVICE_MESH = has_tle_device_mesh()
