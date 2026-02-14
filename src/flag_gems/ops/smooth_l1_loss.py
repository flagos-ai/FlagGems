import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def smooth_l1_elementwise_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    diff = x - y
    ad = tl.abs(diff)

    beta_vec = tl.full(ad.shape, beta, x.dtype)
    loss_beta = 0.5 * diff * diff / beta_vec
    loss_piecewise = tl.where(ad < beta_vec, loss_beta, ad - 0.5 * beta_vec)

    use_piecewise = beta_vec > 0
    loss = tl.where(use_piecewise, loss_piecewise, ad)

    tl.store(out_ptr + offsets, loss, mask=mask)


def _normalize_reduction(reduction):
    if reduction is None:
        return 1
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction code: {reduction}")
    if isinstance(reduction, str):
        mapping = {"none": 0, "mean": 1, "sum": 2}
        key = reduction.lower()
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Invalid reduction: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _prepare_tensors(x, y, dtype=None):
    if x.device != y.device:
        raise ValueError("input and target must be on the same device")
    if dtype is None:
        dtype = torch.result_type(x, y)
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.get_default_dtype()

    if x.device.type != "cuda":
        return None, None, None, None

    bshape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    xb = x.to(dtype).expand(bshape).contiguous()
    yb = y.to(dtype).expand(bshape).contiguous()
    out_buf = torch.empty(bshape, device=x.device, dtype=dtype)
    return xb, yb, out_buf, bshape


def _launch_kernel(xb, yb, out_buf, beta):
    n_elements = out_buf.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    smooth_l1_elementwise_kernel[grid](
        xb, yb, out_buf, n_elements, beta, BLOCK_SIZE=1024
    )


def smooth_l1_loss(self: torch.Tensor, target: torch.Tensor, reduction=1, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)

    prep = _prepare_tensors(self, target)
    if prep[0] is None:
        return torch.ops.aten.smooth_l1_loss(
            self, target, reduction=reduction, beta=beta
        )

    xb, yb, tmp, _ = prep
    _launch_kernel(xb, yb, tmp, float(beta))

    if reduction == 0:
        return tmp
    if reduction == 1:
        return tmp.mean()
    if reduction == 2:
        return tmp.sum()
    raise ValueError(f"Invalid reduction code: {reduction}")


def smooth_l1_loss_out(
    self: torch.Tensor,
    target: torch.Tensor,
    reduction=1,
    beta=1.0,
    *,
    out: torch.Tensor,
):
    logger.debug("GEMS SMOOTH_L1_LOSS_OUT")
    reduction = _normalize_reduction(reduction)

    if self.device.type != "cuda" or target.device.type != "cuda":
        res = torch.ops.aten.smooth_l1_loss(
            self, target, reduction=reduction, beta=beta
        )
        out.copy_(res)
        return out

    xb, yb, tmp, bshape = _prepare_tensors(self, target)
    if xb is None:
        res = torch.ops.aten.smooth_l1_loss(
            self, target, reduction=reduction, beta=beta
        )
        out.copy_(res)
        return out

    _launch_kernel(xb, yb, tmp, float(beta))

    if reduction == 0:
        if out.device != tmp.device or out.dtype != tmp.dtype:
            raise ValueError("out tensor device/dtype mismatch")
        if tuple(out.shape) != tuple(bshape):
            raise ValueError("out tensor shape mismatch for reduction='none'")
        if out.is_contiguous():
            out.copy_(tmp)
        else:
            out.reshape(-1).copy_(tmp.reshape(-1))
        return out

    if reduction == 1:
        res = tmp.mean()
    elif reduction == 2:
        res = tmp.sum()
    else:
        raise ValueError(f"Invalid reduction code: {reduction}")

    if out.numel() != 1:
        raise ValueError("out tensor must have one element for reduced output")
    out.copy_(res)
    return out
