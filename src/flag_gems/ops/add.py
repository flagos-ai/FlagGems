# Comments:
# 
# add(input, other, *, alpha=1) -> Tensor
# 
# Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.
# 
# Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
# :ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.
# 
# Args:
#     A (Tensor or Number): the first operand.
#     B (Tensor or Number): the second operand.
# 
# Keyword arguments:
#     alpha (Number): the multiplier for :attr:`B`.
# 
# Examples::
# 
#     >>> A = torch.tensor([1, 2, 3])
#     >>> B = torch.tensor([4, 5, 6])
#     >>> add(A, B, alpha=2)
#     tensor([ 9, 12, 15])  # 1 + 4*2, 2 + 5*2, 3 + 6*2
# 
#     >>> A = torch.tensor([1, 2, 3])
#     >>> B = 4
#     >>> add(A, B, alpha=2)
#     tensor([ 9, 12, 15])  # 1 + 4*2, 2 + 4*2, 3 + 4*2
# 
#     >>> A = 1
#     >>> B = torch.tensor([4, 5, 6])
#     >>> add(A, B, alpha=2)
#     tensor([ 9, 12, 15])  # 1 + 4*2, 1 + 5*2, 1 + 6*2
# 
#     >>> A = 1
#     >>> B = 4
#     >>> add(A, B, alpha=2)
#     tensor(9)  # 1 + 4*2
# 
# add_(input, other, *, alpha=1) -> Tensor
# 
# In-place version of :func:`~torch.add`. Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input` and stores the result in :attr:`input`.
# 
# Args:
#     A (Tensor or Number): the first operand.
#     B (Tensor or Number): the second operand.
# 
# Keyword arguments:
#     alpha (Number): the multiplier for :attr:`B`.
# 
# Examples::
# 
#     >>> A = torch.tensor([1, 2, 3])
#     >>> B = torch.tensor([4, 5, 6])
#     >>> add_(A, B, alpha=2)
#     tensor([ 9, 12, 15])  # 1 + 4*2, 2 + 5*2, 3 + 6*2
# 
#     >>> A = torch.tensor([1, 2, 3])
#     >>> B = 4
#     >>> add_(A, B, alpha=2)
#     tensor([ 9, 12, 15])  # 1 + 4*2, 2 + 4*2, 3 + 4*2
# 
#     >>> A = 1
#     >>> B = torch.tensor([4, 5, 6])
#     >>> add_(A, B, alpha=2)
#     tensor([ 9, 12, 15])  # 1 + 4*2, 1 + 5*2, 1 + 6*2
# 
#     >>> A = 1
#     >>> B = 4
#     >>> add_(A, B, alpha=2)
#     tensor(9)  # 1 + 4*2

import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def add(A, B, *, alpha=1):
    logger.debug("GEMS ADD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
