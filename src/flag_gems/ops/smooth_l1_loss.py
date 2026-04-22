import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_POINTWISE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
]


@triton.autotune(configs=_POINTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def _smooth_l1_loss_abs_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FLOAT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    if COMPUTE_FLOAT32:
        x = x.to(tl.float32)
        y = y.to(tl.float32)

    out = tl.abs(x - y)
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_POINTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def _smooth_l1_loss_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,
    inv_beta,
    half_beta,
    COMPUTE_FLOAT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    if COMPUTE_FLOAT32:
        x = x.to(tl.float32)
        y = y.to(tl.float32)

    diff = x - y
    abs_diff = tl.abs(diff)
    quadratic = 0.5 * diff * diff * inv_beta
    linear = abs_diff - half_beta
    out = tl.where(abs_diff < beta, quadratic, linear)

    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_POINTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def _smooth_l1_loss_backward_kernel(
    grad_output_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    diff = x - y
    abs_diff = tl.abs(diff)
    beta_val = tl.full(abs_diff.shape, beta, tl.float32)

    neg_one = tl.full(abs_diff.shape, -1.0, tl.float32)
    zero = tl.full(abs_diff.shape, 0.0, tl.float32)
    pos_one = tl.full(abs_diff.shape, 1.0, tl.float32)
    sign = tl.where(diff > 0, pos_one, tl.where(diff < 0, neg_one, zero))

    smooth_grad = tl.where(abs_diff < beta_val, diff / beta_val, sign)
    grad = tl.where(beta_val > 0, smooth_grad, sign)
    tl.store(out_ptr + offsets, (grad_output * grad).to(out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_POINTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def _smooth_l1_loss_backward_reduce_kernel(
    grad_output_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,
    norm,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    diff = x - y
    abs_diff = tl.abs(diff)
    beta_val = tl.full(abs_diff.shape, beta, tl.float32)

    neg_one = tl.full(abs_diff.shape, -1.0, tl.float32)
    zero = tl.full(abs_diff.shape, 0.0, tl.float32)
    pos_one = tl.full(abs_diff.shape, 1.0, tl.float32)
    sign = tl.where(diff > 0, pos_one, tl.where(diff < 0, neg_one, zero))

    smooth_grad = tl.where(abs_diff < beta_val, diff / beta_val, sign)
    grad = tl.where(beta_val > 0, smooth_grad, sign)
    tl.store(
        out_ptr + offsets,
        (grad_output * grad / norm).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _smooth_l1_loss_reduce_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    n_elements,
    beta,
    inv_beta,
    half_beta,
    reduction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)

    diff = x - y
    abs_diff = tl.abs(diff)
    quadratic = 0.5 * diff * diff * inv_beta
    linear = abs_diff - half_beta
    vals = tl.where(abs_diff < beta, quadratic, linear)
    vals = tl.where(mask, vals, 0.0)

    acc = tl.sum(vals, axis=0)
    if reduction == 1:
        acc = acc / n_elements
    tl.store(mid_ptr + pid, acc)


@triton.jit
def _smooth_l1_loss_abs_reduce_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    n_elements,
    reduction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)

    vals = tl.abs(x - y)
    vals = tl.where(mask, vals, 0.0)

    acc = tl.sum(vals, axis=0)
    if reduction == 1:
        acc = acc / n_elements
    tl.store(mid_ptr + pid, acc)


@triton.jit
def _smooth_l1_loss_finalize_kernel(mid_ptr, out_ptr, mid_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < mid_size
    vals = tl.load(mid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(vals, axis=0))


def _normalize_reduction(reduction):
    if reduction is None:
        return "mean"
    if isinstance(reduction, str):
        reduction = reduction.lower()
        if reduction in ("none", "mean", "sum"):
            return reduction
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        mapping = {0: "none", 1: "mean", 2: "sum"}
        if reduction in mapping:
            return mapping[reduction]
        raise ValueError(f"Invalid reduction code: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _reduction_code(reduction):
    normalized = _normalize_reduction(reduction)
    return {"none": 0, "mean": 1, "sum": 2}[normalized]


def _prepare_tensor(tensor, dtype):
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.view(-1)


def smooth_l1_loss(inp, target, reduction="mean", beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)

    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    if inp.device.type != "cuda" or target.device.type != "cuda":
        return torch.nn.functional.smooth_l1_loss(
            inp, target, reduction=reduction, beta=beta
        )

    if inp.shape == target.shape:
        x, y = inp, target
        output_shape = inp.shape
    else:
        x, y = torch.broadcast_tensors(inp, target)
        output_shape = x.shape
    out_dtype = torch.result_type(x, y)
    if not out_dtype.is_floating_point:
        out_dtype = torch.get_default_dtype()

    x = _prepare_tensor(x, out_dtype)
    y = _prepare_tensor(y, out_dtype)
    n_elements = x.numel()
    if n_elements == 0:
        if reduction == "none":
            return torch.empty(output_shape, device=x.device, dtype=out_dtype)
        if reduction == "sum":
            return torch.zeros((), device=x.device, dtype=out_dtype)
        return torch.full((), float("nan"), device=x.device, dtype=out_dtype)

    compute_fp32 = out_dtype != torch.float16

    if reduction == "none":
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if beta == 0.0:
            _smooth_l1_loss_abs_kernel[grid](
                x,
                y,
                out,
                n_elements,
                COMPUTE_FLOAT32=compute_fp32,
            )
        else:
            _smooth_l1_loss_kernel[grid](
                x,
                y,
                out,
                n_elements,
                float(beta),
                1.0 / float(beta),
                0.5 * float(beta),
                COMPUTE_FLOAT32=compute_fp32,
            )
        return out.view(output_shape)

    # Large reductions benefit from avoiding a full-sized temporary output tensor.
    # Keep the float16 path narrowly scoped to the largest workloads so we do not
    # regress the smaller cases that are already near parity.
    use_fused_reduction = (
        (out_dtype == torch.float32 and n_elements > (1 << 20))
        or (out_dtype == torch.float16 and n_elements >= (1 << 24))
    )
    if use_fused_reduction:
        reduction_code = _reduction_code(reduction)
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        mid_size = triton.cdiv(n_elements, block_size)
        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty((), dtype=torch.float32, device=x.device)

        if beta == 0.0:
            _smooth_l1_loss_abs_reduce_kernel[(mid_size,)](
                x,
                y,
                mid,
                n_elements,
                reduction_code,
                BLOCK_SIZE=block_size,
            )
        else:
            _smooth_l1_loss_reduce_kernel[(mid_size,)](
                x,
                y,
                mid,
                n_elements,
                float(beta),
                1.0 / float(beta),
                0.5 * float(beta),
                reduction_code,
                BLOCK_SIZE=block_size,
            )
        _smooth_l1_loss_finalize_kernel[(1,)](
            mid,
            out,
            mid_size,
            BLOCK_SIZE=triton.next_power_of_2(mid_size),
        )
        return out.to(dtype=out_dtype)

    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if beta == 0.0:
        _smooth_l1_loss_abs_kernel[grid](
            x,
            y,
            out,
            n_elements,
            COMPUTE_FLOAT32=compute_fp32,
        )
    else:
        _smooth_l1_loss_kernel[grid](
            x,
            y,
            out,
            n_elements,
            float(beta),
            1.0 / float(beta),
            0.5 * float(beta),
            COMPUTE_FLOAT32=compute_fp32,
        )
    if reduction == "sum":
        return out.sum()
    return out.mean()


def smooth_l1_loss_backward(grad_output, self, target, reduction, beta):
    logger.debug("GEMS SMOOTH_L1_LOSS_BACKWARD")
    beta = float(beta)
    reduction = _reduction_code(reduction)

    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    if (
        grad_output.device.type != "cuda"
        or self.device.type != "cuda"
        or target.device.type != "cuda"
    ):
        return torch.ops.aten.smooth_l1_loss_backward(
            grad_output, self, target, reduction, beta
        )

    self_b, target_b = torch.broadcast_tensors(self, target)
    if self_b.numel() == 0:
        return torch.empty_like(self)

    x = self_b.contiguous()
    y = target_b.contiguous()
    n_elements = x.numel()
    grad_input = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if reduction == 0:
        grad_output_b, _, _ = torch.broadcast_tensors(grad_output, self_b, target_b)
        _smooth_l1_loss_backward_kernel[grid](
            grad_output_b.contiguous(),
            x,
            y,
            grad_input,
            n_elements,
            float(beta),
        )
    else:
        norm = float(n_elements) if reduction == 1 else 1.0
        grad_output_scalar = grad_output.contiguous().view(-1)
        _smooth_l1_loss_backward_reduce_kernel[grid](
            grad_output_scalar,
            x,
            y,
            grad_input,
            n_elements,
            float(beta),
            norm,
        )

    if grad_input.shape != self.shape:
        grad_input = grad_input.sum_to_size(self.shape)
    return grad_input
