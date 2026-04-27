import importlib.util
from importlib import metadata


def SkipVersion(module_name, pattern):
    if importlib.util.find_spec(module_name) is None:
        return True

    op = pattern[0]
    assert op in ("=", "<", ">"), f"Invalid comparison operator: {op}"
    try:
        M, N = pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise ValueError("Cannot parse version number from pattern.")

    try:
        version = metadata.version(module_name)
        major, minor = map(int, version.split(".")[:2])
    except Exception:
        raise ImportError(f"Cannot determine version of module: {module_name}")

    if op == "=":
        return major == M and minor == N

    if op == "<":
        return (major, minor) < (M, N)

    return (major, minor) > (M, N)
