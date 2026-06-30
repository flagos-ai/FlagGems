# import logging

# import torch
# import triton
# import triton.language as tl

# logger = logging.getLogger(__name__)

# configs = [
#     # triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 8196}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 16384}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 32768}, num_warps=4),
#     triton.Config({"BLOCK_SIZE": 65536}, num_warps=4),
#     # triton.Config({"BLOCK_SIZE": 131072}, num_warps=4),
# ]


# @triton.autotune(configs=configs, key=["N"])
# @triton.jit
# def greater_kernel(x, y, out, N, BLOCK_SIZE: tl.constexpr):
#     offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < N
#     xv = tl.load(x + offsets, mask=mask)
#     yv = tl.load(y + offsets, mask=mask)
#     tl.store(out + offsets, xv > yv, mask=mask)


# @triton.autotune(configs=configs, key=["N"])
# @triton.jit
# def greater_scalar_kernel(x, y, out, N, BLOCK_SIZE: tl.constexpr):
#     offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < N
#     xv = tl.load(x + offsets, mask=mask).to(tl.float32)
#     tl.store(out + offsets, xv > y, mask=mask)


# def greater(A, B):
#     logger.debug("GEMS_KUNLUNXIN GREATER")
#     A, B = A.contiguous(), B.contiguous()
#     out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
#     N = A.numel()
#     if N > 0:
#         grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
#         greater_kernel[grid](A, B, out, N)
#     return out


# def greater_out(A, B, *, out=None):
#     logger.debug("GEMS_KUNLUNXIN GREATER_OUT")
#     if out is None:
#         return greater(A, B)
#     A, B = A.contiguous(), B.contiguous()
#     N = A.numel()
#     if N > 0:
#         grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
#         greater_kernel[grid](A, B, out, N)
#     return out


# def greater_scalar(A, B):
#     logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR")
#     A = A.contiguous()
#     out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
#     N = A.numel()
#     if N > 0:
#         grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
#         greater_scalar_kernel[grid](A, B, out, N)
#     return out


# def greater_scalar_out(A, B, *, out=None):
#     logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
#     if out is None:
#         return greater_scalar(A, B)
#     A = A.contiguous()
#     N = A.numel()
#     if N > 0:
#         grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
#         greater_scalar_kernel[grid](A, B, out, N)
#     return out


import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_func(x, y):
    return x > y


def greater(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER")
    return greater_func(A, B)


def greater_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_OUT")
    if out is None:
        return greater_func(A, B)
    greater_func(A, B, out0=out)
    return out


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_func_scalar(x, y):
    return x.to(tl.float32) > y


def greater_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR")
    return greater_func_scalar(A, B)


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    if out is None:
        return greater_func_scalar(A, B)
    greater_func_scalar(A, B, out0=out)
    return out
