import triton

if triton.__version__ >= "3.4":
  from .mm import mm

__all__ = ["*"]

