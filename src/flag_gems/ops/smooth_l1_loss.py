import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


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


def _reduction_to_aten_int(reduction_s: str) -> int:
    return {"none": 0, "mean": 1, "sum": 2}[reduction_s]


def _launch_smooth_l1_elementwise(x, y, out_buf, beta):
    n_elements = out_buf.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        smooth_l1_elementwise_kernel[grid](
            x, y, out_buf, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE
        )


def _prepare_tensors_for_elementwise(x, y, dtype=None):
    import flag_gems as _fg

    if dtype is None:
        dtype = torch.result_type(x, y)
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.get_default_dtype()
    if x.device != y.device:
        raise ValueError("input and target must be on the same device")
    if x.device.type != _fg.device:
        return None, None, None, None

    bshape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    xb = x.to(dtype).expand(bshape).contiguous()
    yb = y.to(dtype).expand(bshape).contiguous()
    out_buf = torch.empty(bshape, device=x.device, dtype=dtype)
    return xb, yb, out_buf, bshape


def _aten_fallback(x, y, reduction_s, beta):
    return torch.ops.aten.smooth_l1_loss.default(
        x,
        y,
        reduction=_reduction_to_aten_int(reduction_s),
        beta=beta,
    )


def smooth_l1_loss(inp, target, reduction=1, beta=1.0):
    logger.debug("GEMS SMOOTH L1 LOSS")
    reduction_s = _normalize_reduction(reduction)
    beta = float(beta)

    prep = _prepare_tensors_for_elementwise(inp, target)
    if prep[0] is None:
        return _aten_fallback(inp, target, reduction_s, beta)

    xb, yb, tmp, _ = prep
    _launch_smooth_l1_elementwise(xb, yb, tmp, beta)

    if reduction_s == "none":
        return tmp
    if reduction_s == "mean":
        return tmp.mean()
    if reduction_s == "sum":
        return tmp.sum()
    raise ValueError(f"Invalid reduction: {reduction_s}")


def smooth_l1_loss_out(inp, target, reduction=1, beta=1.0, *, out):
    logger.debug("GEMS SMOOTH L1 LOSS OUT")
    reduction_s = _normalize_reduction(reduction)
    beta = float(beta)

    prep = _prepare_tensors_for_elementwise(inp, target)
    if prep[0] is None:
        return torch.ops.aten.smooth_l1_loss.out(
            inp,
            target,
            _reduction_to_aten_int(reduction_s),
            beta,
            out=out,
        )

    xb, yb, tmp, bshape = prep
    _launch_smooth_l1_elementwise(xb, yb, tmp, beta)

    if reduction_s == "none":
        if out.device != tmp.device:
            raise ValueError("out tensor device mismatch")
        if out.dtype != tmp.dtype:
            raise ValueError("out tensor dtype mismatch")
        if tuple(out.shape) != tuple(bshape):
            raise ValueError("out tensor shape mismatch for reduction='none'")
        if out.is_contiguous():
            out.copy_(tmp)
        else:
            out.reshape(-1).copy_(tmp.reshape(-1))
        return out

    if reduction_s == "mean":
        res = tmp.mean()
    elif reduction_s == "sum":
        res = tmp.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction_s}")

    if out.device != res.device:
        raise ValueError("out tensor device mismatch")
    if out.dtype != res.dtype:
        raise ValueError("out tensor dtype mismatch")
    if out.numel() != 1:
        raise ValueError("out tensor must have one element for reduced output")
    out.copy_(res)
    return out
