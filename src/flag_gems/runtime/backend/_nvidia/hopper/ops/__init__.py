import triton

if triton.__version__ >= "3.4":
    from .mm import mm, mm_out  # noqa: F401

__all__ = ["*"]
