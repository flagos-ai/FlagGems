import triton

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy(src):
    return src
