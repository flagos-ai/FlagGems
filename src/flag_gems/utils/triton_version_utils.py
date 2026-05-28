import re

import triton
from packaging.version import InvalidVersion, Version


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


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    if not _triton_version_at_least(major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language as _tle  # noqa: F401

        return True
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
