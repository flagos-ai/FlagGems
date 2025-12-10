# # abs(A) computes the absolute value of each element in the input tensor A using the abs_func function.
# 
# # Args:
# #     A (Tensor): The input tensor.
# 
# # Keyword arguments:
# #     None
# 
# # Examples:
# #     >>> A = torch.tensor([-1, 2, -3, 4])
# #     >>> abs(A)
# #     tensor([1, 2, 3, 4])

# Comments:
# 
# abs(A) computes the absolute value of each element in the input tensor A using the abs_func function.
# 
# Args:
#     A (Tensor): The input tensor.
# 
# Keyword arguments:
#     None
# 
# Examples:
#     >>> A = torch.tensor([-1, 2, -3, 4])
#     >>> abs(A)
#     tensor([1, 2, 3, 4])

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    logger.debug("GEMS ABS")
    return abs_func(A)


def abs_(A):
    logger.debug("GEMS ABS_")
    abs_func(A, out0=A)
    return A
