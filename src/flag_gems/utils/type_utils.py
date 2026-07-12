import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes


def type_promotion(*args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND):
    computation_dtype, result_dtype = elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    return computation_dtype, result_dtype


_accumulator_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def get_accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    return _accumulator_dtype_map.get(dtype, dtype)
