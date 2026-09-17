import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic as xpu_pointwise_dynamic

logger = logging.getLogger(__name__)


UNROLL_NUM = 2
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False

FLAT_MAX_NUMEL = 4 * 1024 * 1024

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@xpu_pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def log_sigmoid_backward_func(grad_output, self):
    go = grad_output.to(tl.float32)
    x = self.to(tl.float32)
    return (go * tl.sigmoid(0.0 - x)).to(grad_output.dtype)


def _pick_block(n_elements):
    if n_elements >= 32768 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


@triton.jit
def log_sigmoid_backward_flat_kernel(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    g = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(self_ptr + offsets, mask=mask)
    derivative = 1.0 / (1.0 + tl.exp(x.to(tl.float32)))
    res = g.to(tl.float32) * derivative
    tl.store(
        grad_input_ptr + offsets,
        res.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def log_sigmoid_backward_flat_kernel_unmasked(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g = tl.load(grad_output_ptr + offsets)
    x = tl.load(self_ptr + offsets)
    derivative = 1.0 / (1.0 + tl.exp(x.to(tl.float32)))
    res = g.to(tl.float32) * derivative
    tl.store(grad_input_ptr + offsets, res.to(grad_input_ptr.dtype.element_ty))


def _can_use_flat_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (
            grad_input is None
            or (
                grad_input.shape == self.shape
                and grad_input.dtype == self.dtype
                and grad_input.is_contiguous()
            )
        )
    )


def _launch_flat_kernel(grad_output, self, grad_input):
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        log_sigmoid_backward_flat_kernel[grid](
            grad_output,
            self,
            grad_input,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        log_sigmoid_backward_flat_kernel_unmasked[grid](
            grad_output,
            self,
            grad_input,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")
    if _can_use_flat_kernel(grad_output, self):
        if self.numel() > FLAT_MAX_NUMEL:
            return log_sigmoid_backward_func(grad_output, self)
        return _launch_flat_kernel(grad_output, self, torch.empty_like(self))
    return log_sigmoid_backward_func(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD OUT")
    if _can_use_flat_kernel(grad_output, self, grad_input):
        if self.numel() > FLAT_MAX_NUMEL:
            return log_sigmoid_backward_func(grad_output, self, out0=grad_input)
        return _launch_flat_kernel(grad_output, self, grad_input)
    return log_sigmoid_backward_func(grad_output, self, out0=grad_input)
