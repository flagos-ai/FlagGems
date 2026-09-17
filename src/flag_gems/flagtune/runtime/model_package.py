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

"""Resolve whether the active platform has a usable FlagTune model package."""

from __future__ import annotations

import os
from typing import Any, Optional

_MODEL_SOURCE_ENVIRONMENT = (
    "HOME",
    "FLAGTUNE_MODEL_DIR",
    "FLAGTUNE_MODEL_CACHE",
    "FLAGTUNE_LOCAL_MANIFEST",
    "FLAGTUNE_MODEL_BASE_URL",
    "FLAGTUNE_MODEL_VERSION",
    "FLAGTUNE_MODEL_DOWNLOAD_LATEST",
    "FLAGTUNE_DISABLE_REMOTE",
)
_MODEL_MANAGER: Optional[Any] = None
_PLATFORM_PACKAGE_AVAILABILITY = {}


def _get_model_manager():
    """Return the lazy manager used only for platform-package resolution."""
    global _MODEL_MANAGER
    if _MODEL_MANAGER is None:
        from triton.flagtune.runtime.model_loader import FlagTuneModelManager

        _MODEL_MANAGER = FlagTuneModelManager()
    return _MODEL_MANAGER


def _discover_platform_key() -> str:
    """Return the canonical platform key for the active Triton device."""
    from triton.flagtune.contract.identity import discover_gpu_metadata

    return str(discover_gpu_metadata()["platform_key"])


def _availability_cache_key(platform_key: str):
    """Scope a process result to every setting that selects a model source."""
    return (
        platform_key,
        *(os.environ.get(name) for name in _MODEL_SOURCE_ENVIRONMENT),
    )


def _is_unversioned_platform_miss(exc: FileNotFoundError, platform_key: str) -> bool:
    """Recognize only a Manifest with no entry for the current platform."""
    prefix = f"FlagTune Manifest has no package for platform {platform_key!r};"
    return str(exc).startswith(prefix)


def platform_model_package_available() -> bool:
    """Return whether the active platform resolves to an outer model package.

    Resolution follows FlagTree's existing user-directory, package-cache,
    Manifest, and download rules. Only an unversioned Manifest miss means the
    platform is unadapted. Fixed-version misses, disabled remote access,
    malformed Manifests, download failures, checksum mismatches, and invalid
    archives continue to raise their original exceptions.
    """
    platform_key = _discover_platform_key()
    cache_key = _availability_cache_key(platform_key)
    cached = _PLATFORM_PACKAGE_AVAILABILITY.get(cache_key)
    if cached is not None:
        return cached

    manager = _get_model_manager()
    try:
        manager.resolve(
            "flaggems/platform-package-probe",
            "availability",
            platform_key=platform_key,
            dtype_key="f32",
        )
    except FileNotFoundError as exc:
        if not _is_unversioned_platform_miss(exc, platform_key):
            raise
        available = False
    else:
        available = True

    _PLATFORM_PACKAGE_AVAILABILITY[cache_key] = available
    return available


__all__ = ["platform_model_package_available"]
