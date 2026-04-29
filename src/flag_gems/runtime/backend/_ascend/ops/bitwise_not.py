import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def bitwise_not_func(x):
    return ~x

def bitwise_not(A):
    print("ASCEND GEMS BITWISE NOT")
    return bitwise_not_func(A)

def bitwise_not_(A):
    print("ASCEND GEMS BITWISE NOT_")
    bitwise_not_func(A, out0=A)
    return A