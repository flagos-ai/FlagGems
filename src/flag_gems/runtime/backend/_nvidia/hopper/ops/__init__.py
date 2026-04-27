import triton

if triton.__version__ >= "3.4":
    from .fill import (  # noqa: F401
        fill_scalar,
        #fill_scalar_,
        fill_scalar_out,
        fill_tensor,
        fill_tensor_,
        fill_tensor_out,
    )
    from .mm import mm, mm_out  # noqa: F401
    from .mul import mul  # noqa: F401
    from .sqrt import sqrt, sqrt_  # noqa: F401
    from .w8a8_block_fp8_matmul import w8a8_block_fp8_matmul  # noqa: F401

__all__ = ["*"]
