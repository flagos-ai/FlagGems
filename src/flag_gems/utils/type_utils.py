import torch._prims_common as utils


def type_promotion(*args, type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND):
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    return computation_dtype, result_dtype
