import triton.language as tl

__all__ = ["erf", "exp"]


def __getattr__(name):
    if name == "erf":
        return tl.erf
    elif name == "exp":
        return tl.exp
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
