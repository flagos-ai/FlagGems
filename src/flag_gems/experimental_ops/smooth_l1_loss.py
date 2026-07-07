import torch
import triton
import triton.language as tl


@triton.jit
def smooth_l1_elementwise_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,  # scalar
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    diff = x - y
    ad = tl.abs(diff)

    # Broadcast beta to vector shape
    beta_vec = tl.full(ad.shape, beta, x.dtype)

    # Smooth L1 piecewise (for beta > 0):
    # 0.5 * x^2 / beta             if |x| < beta
    # |x| - 0.5 * beta             otherwise
    loss_beta = 0.5 * diff * diff / beta_vec
    loss_piecewise = tl.where(ad < beta_vec, loss_beta, ad - 0.5 * beta_vec)

    # If beta <= 0, fall back to L1: |x|
    # Use vectorized condition to avoid divide-by-zero
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


def _parse_smooth_l1_args(args, kwargs, out_variant=False):
    if len(args) < 2:
        raise TypeError("smooth_l1_loss requires at least input and target tensors")

    x = args[0]
    y = args[1]

    beta = kwargs.pop("beta", None)
    reduction = kwargs.pop("reduction", None)
    out = kwargs.pop("out", None) if out_variant else None

    # Parse remaining positional arguments flexibly
    rest = list(args[2:])

    # Try to infer reduction and beta from positional args
    def maybe_set_reduction(val):
        nonlocal reduction
        if reduction is not None:
            return False
        if isinstance(val, str):
            reduction = val
            return True
        if isinstance(val, int) and val in (0, 1, 2):
            reduction = val
            return True
        return False

    def maybe_set_beta(val):
        nonlocal beta
        if beta is not None:
            return False
        if isinstance(val, (float, int)):
            beta = float(val)
            return True
        return False

    # Accept either order for the two optional parameters
    for val in rest:
        if not maybe_set_reduction(val):
            maybe_set_beta(val)

    if beta is None:
        beta = 1.0
    reduction = _normalize_reduction(reduction)

    return x, y, reduction, float(beta), out, kwargs


def _launch_smooth_l1_elementwise(x, y, out_buf, beta):
    n_elements = out_buf.numel()
    if n_elements == 0:
        return  # nothing to do

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    smooth_l1_elementwise_kernel[grid](
        x, y, out_buf, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE
    )


def _prepare_tensors_for_elementwise(x, y, dtype=None):
    if dtype is None:
        dtype = torch.result_type(x, y)
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.get_default_dtype()
    device = x.device
    if x.device != y.device:
        raise ValueError("input and target must be on the same device")
    if not device.type == "cuda":
        return None, None, None, None  # signal fallback

    # Broadcast to a common shape
    bshape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    xb = x.to(dtype).expand(bshape).contiguous()
    yb = y.to(dtype).expand(bshape).contiguous()
    out_buf = torch.empty(bshape, device=device, dtype=dtype)
    return xb, yb, out_buf, bshape


def smooth_l1_loss(*args, **kwargs):
    x, y, reduction, beta, _, leftover = _parse_smooth_l1_args(
        args, kwargs, out_variant=False
    )
    if leftover:
        raise TypeError(f"Unexpected keyword arguments: {list(leftover.keys())}")

    prep = _prepare_tensors_for_elementwise(x, y)
    if prep[0] is None:
        # Fallback to PyTorch if not CUDA
        return torch.ops.aten.smooth_l1_loss(x, y, reduction=reduction, beta=beta)

    xb, yb, tmp, _ = prep
    _launch_smooth_l1_elementwise(xb, yb, tmp, beta)

    if reduction == "none":
        return tmp
    elif reduction == "mean":
        return tmp.mean()
    elif reduction == "sum":
        return tmp.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def smooth_l1_loss_out(*args, **kwargs):
    x, y, reduction, beta, out, leftover = _parse_smooth_l1_args(
        args, kwargs, out_variant=True
    )
    if leftover:
        raise TypeError(f"Unexpected keyword arguments: {list(leftover.keys())}")

    # Fallback if not CUDA
    if x.device.type != "cuda" or y.device.type != "cuda":
        res = torch.ops.aten.smooth_l1_loss(x, y, reduction=reduction, beta=beta)
        if out is None:
            return res
        else:
            out.copy_(res)
            return out

    xb, yb, tmp, bshape = _prepare_tensors_for_elementwise(x, y)
    if xb is None:
        # Should not happen due to device check above
        res = torch.ops.aten.smooth_l1_loss(x, y, reduction=reduction, beta=beta)
        if out is None:
            return res
        else:
            out.copy_(res)
            return out

    _launch_smooth_l1_elementwise(xb, yb, tmp, beta)

    if reduction == "none":
        if out is None:
            return tmp
        # Validate 'out' shape/device/dtype for 'none'
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
    else:
        if reduction == "mean":
            res = tmp.mean()
        elif reduction == "sum":
            res = tmp.sum()
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        if out is None:
            return res
        # For reduced results, expect out to be a scalar tensor (numel == 1)
        if out.device != res.device:
            raise ValueError("out tensor device mismatch")
        if out.dtype != res.dtype:
            raise ValueError("out tensor dtype mismatch")
        if out.numel() != 1:
            raise ValueError("out tensor must have one element for reduced output")
        out.copy_(res)
        return out
