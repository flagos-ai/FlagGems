# Comments:
# 
# ```
# addcdiv(inp, tensor1, tensor2, value=1.0, out=None) -> Tensor
# 
# Performs element-wise addition where each element of the input tensor is added to the product of the corresponding elements of tensor1 and value divided by tensor2.
# 
# .. math::
#     \text{{out}}_i = \text{{inp}}_i + \text{{value}} \times \frac{\text{{tensor1}}_i}{\text{{tensor2}}_i}
# 
# Supports broadcasting, type promotion, and can take tensors or scalars as inputs for tensor1 and tensor2. The value is a scalar multiplier.
# 
# Args:
#     inp (Tensor): The input tensor.
#     tensor1 (Tensor or Number): The tensor or number to be divided by tensor2 and multiplied by value.
#     tensor2 (Tensor or Number): The tensor or number to divide tensor1 by.
# 
# Keyword arguments:
#     value (Number): The scalar multiplier for the division result of tensor1 and tensor2.
#     out (Tensor, optional): The output tensor. If not provided, a new tensor is created.
# 
# Examples::
#     >>> inp = torch.tensor([1., 2., 3.])
#     >>> tensor1 = torch.tensor([4., 5., 6.])
#     >>> tensor2 = torch.tensor([2., 1., 3.])
#     >>> value = 2.0
#     >>> addcdiv(inp, tensor1, tensor2, value)
#     tensor([3., 12., 9.])
# ```

import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcdiv_kernel(x, t1, t2, value):
    return x + value * (t1 / t2)


def addcdiv(inp, tensor1, tensor2, value=1.0, out=None):
    logger.debug("GEMS ADDCDIV FORWARD")

    if out is None:
        out = torch.empty_like(inp)

    addcdiv_kernel(inp, tensor1, tensor2, value, out0=out)

    return out
