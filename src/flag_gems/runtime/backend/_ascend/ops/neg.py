import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def neg_func(x):
    return -x

def neg(A):
    print("ASCEND GEMS NEG")
    return neg_func(A)

def neg_(A):
    print("ASCEND GEMS NEG_")
    return neg_func(A, out0=A)