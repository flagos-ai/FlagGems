import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

_tanh = tl_extra_shim.tanh
logger = logging.getLogger(__name__)

_AUTOTUNE_CFGS = [
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
]


@libentry()
@triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
@triton.jit
def _tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Promote fp16/bf16 to fp32 then cast back. Uses the platform's
    # hardware tanh via tl_extra_shim for parity with the existing
    # FlagGems implementation.
    y = _tanh(x.to(tl.float32)).to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


@libentry()
@triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
@triton.jit
def _tanh_backward_kernel(
    y_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    y_raw = tl.load(y_ptr + offs, mask=mask, other=0.0)
    dy_raw = tl.load(dy_ptr + offs, mask=mask, other=0.0)
    y = y_raw.to(tl.float32)
    dy = dy_raw.to(tl.float32)
    dx = dy * (1.0 - y * y)
    tl.store(dx_ptr + offs, dx.to(y_raw.dtype), mask=mask)


def _launch_fwd(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = inp.numel()
    if n == 0:
        return out
    if not inp.is_contiguous():
        inp = inp.contiguous()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _tanh_kernel[grid](inp, out, n)
    return out


def tanh(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS TANH FORWARD")
    if not self.is_floating_point():
        self = self.to(torch.float32)
    return _launch_fwd(self, torch.empty_like(self))


def tanh_(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS TANH_ FORWARD")
    if not A.is_floating_point():
        raise RuntimeError("tanh_: in-place op requires a floating-point tensor")
    return _launch_fwd(A, A)


def tanh_backward(grad_output: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS TANH BACKWARD")
    if not output.is_contiguous():
        output = output.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    dx = torch.empty_like(output)
    n = output.numel()
    if n == 0:
        return dx
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _tanh_backward_kernel[grid](output, grad_output, dx, n)
    return dx
