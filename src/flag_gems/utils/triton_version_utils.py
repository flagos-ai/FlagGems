import triton


def _triton_version_at_least(major: int, minor: int, patch: int = 0) -> bool:
    version = str(getattr(triton, "__version__", "0.0.0")).split("+", 1)[0]
    parts = version.split(".")
    parsed = []
    for part in parts[:3]:
        digits = []
        for ch in part:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        parsed.append(int("".join(digits)) if digits else 0)
    while len(parsed) < 3:
        parsed.append(0)
    return tuple(parsed) >= (major, minor, patch)


HAS_TLE = False
if _triton_version_at_least(3, 1, 0):
    try:
        import triton.experimental.tle.language as _tle  # noqa: F401

        HAS_TLE = True
    except ImportError:
        pass
