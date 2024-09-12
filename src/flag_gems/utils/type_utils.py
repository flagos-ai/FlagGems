from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes


def type_promotion(*args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND):
    computation_dtype, result_dtype = elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    return computation_dtype, result_dtype
