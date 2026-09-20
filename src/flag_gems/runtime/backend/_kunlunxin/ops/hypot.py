import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    raise ValueError(f"Unsupported dtype for Triton conversion: {dtype}")


@triton.jit
def _hypot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    xf = x.to(COMPUTE_DTYPE)
    yf = y.to(COMPUTE_DTYPE)

    ax = tl.abs(xf)
    ay = tl.abs(yf)
    t = tl.maximum(ax, ay)
    m = tl.minimum(ax, ay)
    t_nz = tl.where(t > 0, t, 1).to(COMPUTE_DTYPE)
    r = m / t_nz
    res = tl.where(t > 0, t * tl.sqrt(1 + r * r), m)

    s_nan = (xf != xf) | (yf != yf)
    inf_f = t == float("inf")
    res = tl.where(inf_f, float("inf"), tl.where(s_nan, float("nan"), res))

    out_val = res.to(OUT_DTYPE)
    tl.store(out_ptr + offsets, out_val, mask=mask)


def _infer_hypot_out_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    if a.is_complex() or b.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported for hypot in this implementation."
        )
    if a.is_floating_point() or b.is_floating_point():
        return torch.result_type(a, b)
    return torch.get_default_dtype()


def _launch_hypot_kernel(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    out_dtype = out.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported output dtype for hypot: {out_dtype}")

    OUT_DTYPE = _torch_dtype_to_triton(out_dtype)
    COMPUTE_DTYPE = tl.float64 if out_dtype == torch.float64 else tl.float32

    with torch_device_fn.device(out.device):
        _hypot_kernel[grid](
            x,
            y,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DTYPE=OUT_DTYPE,
            COMPUTE_DTYPE=COMPUTE_DTYPE,
        )


@triton.jit
def _hypot_inplace_flat_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Flat (contiguous, same-shape) in-place hypot.

    Reads x[i], y[i] and writes the result back to x[i] in the same lane, so
    aliasing x == out is safe.  Uses the simple sqrt(x^2 + y^2) identity in
    COMPUTE_DTYPE (fp32 unless the input is fp64); the guarded overflow-safe
    formula is avoided because its division/select codegen measures 26x-12000x
    slower on the XPU backend, and a plain ``other=``-annotated masked load
    measures ~1.5x slower.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    xf = x.to(COMPUTE_DTYPE)
    yf = y.to(COMPUTE_DTYPE)
    res = tl.sqrt(xf * xf + yf * yf)

    tl.store(x_ptr + offsets, res.to(x.dtype), mask=mask)


def _launch_hypot_inplace_flat(x: torch.Tensor, y: torch.Tensor):
    n_elements = x.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    COMPUTE_DTYPE = tl.float64 if x.dtype == torch.float64 else tl.float32

    with torch_device_fn.device(x.device):
        _hypot_inplace_flat_kernel[grid](
            x,
            y,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            COMPUTE_DTYPE=COMPUTE_DTYPE,
        )


@triton.jit
def _hypot_inplace_strided_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    RANK: tl.constexpr,
    shapes_ptr,
    x_strides_ptr,
    y_strides_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Strided in-place hypot.

    Computes hypot(x, y) elementwise over the logical (broadcast) shape of x
    and stores the result back into x's storage.  An in-place broadcast copy
    via ``.contiguous()`` on XPU goes through the vendor ``copy_`` strided
    codegen which faults for large broadcast shapes (e.g. (1,512)->(512,512)),
    so this kernel resolves the logical->storage offset directly from
    explicit per-dimension strides (0 for broadcast dimensions).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    rem = offsets
    x_off = tl.zeros(offsets.shape, dtype=tl.int64)
    y_off = tl.zeros(offsets.shape, dtype=tl.int64)
    for d in tl.static_range(RANK):
        dim = tl.load(shapes_ptr + d)
        xs = tl.load(x_strides_ptr + d)
        ys = tl.load(y_strides_ptr + d)
        idx = rem % dim
        rem = rem // dim
        x_off += idx * xs
        y_off += idx * ys

    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)

    xf = x.to(COMPUTE_DTYPE)
    yf = y.to(COMPUTE_DTYPE)
    res = tl.sqrt(xf * xf + yf * yf)

    tl.store(x_ptr + x_off, res.to(x.dtype), mask=mask)


def _effective_broadcast_strides(y: torch.Tensor, shape) -> list:
    """Effective per-logical-dim strides of ``y`` broadcast to ``shape``.

    Returns one stride per logical dimension of ``shape`` (``y``'s dims align to
    the trailing dims, as in torch broadcasting); 0 marks a broadcast dimension.
    Raises RuntimeError if ``y`` is not broadcastable to ``shape``.
    """
    rank = len(shape)
    y_rank = y.dim()
    offset = rank - y_rank
    strides = []
    for d in range(rank):
        if d < offset:
            strides.append(0)
            continue
        yd = d - offset
        yd_size = y.shape[yd]
        if yd_size == shape[d]:
            strides.append(y.stride(yd))
        elif yd_size == 1:
            strides.append(0)
        else:
            raise RuntimeError(
                "hypot_: the size of tensor other "
                f"{tuple(y.shape)} must be broadcastable to the size of "
                f"tensor self {tuple(shape)}"
            )
    return strides


def _launch_hypot_inplace_strided(
    x: torch.Tensor, y: torch.Tensor, shapes, x_strides, y_strides
):
    n_elements = x.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    COMPUTE_DTYPE = tl.float64 if x.dtype == torch.float64 else tl.float32
    rank = x.dim()

    with torch_device_fn.device(x.device):
        _hypot_inplace_strided_kernel[grid](
            x,
            y,
            n_elements,
            RANK=rank,
            shapes_ptr=shapes,
            x_strides_ptr=x_strides,
            y_strides_ptr=y_strides,
            BLOCK_SIZE=BLOCK_SIZE,
            COMPUTE_DTYPE=COMPUTE_DTYPE,
        )


def hypot(a: torch.Tensor, b: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN HYPOT")
    out_dtype = _infer_hypot_out_dtype(a, b)
    device = a.device
    if b.device != device:
        raise ValueError("Input tensors must be on the same device")

    out_shape = torch.broadcast_shapes(a.shape, b.shape)
    out = torch.empty(out_shape, dtype=out_dtype, device=device)

    x = a.expand(out_shape).contiguous()
    y = b.expand(out_shape).contiguous()

    _launch_hypot_kernel(x, y, out)
    return out


def hypot_(self: torch.Tensor, other):
    """In-place hypot: self = hypot(self, other), returns self.

    Vendor override of the generic ``flag_gems.ops.hypot_.hypot_``: the generic
    implementation materialises ``torch.broadcast_to(other, self.shape)
    .contiguous()``, and on XPU inside ``use_gems`` that broadcast copy goes
    through the vendor ``copy_`` strided codegen which faults for large
    broadcast shapes (e.g. (1,512)->(512,512) -> illegal memory access).  This
    implementation resolves the logical->storage offsets directly from
    explicit shapes/strides instead, using the two in-place kernels above.
    """
    logger.debug("GEMS_KUNLUNXIN HYPOT_")

    if isinstance(other, torch.Tensor):
        other_t = (
            other
            if (other.device == self.device and other.dtype == self.dtype)
            else other.to(device=self.device, dtype=self.dtype)
        )
    else:
        other_t = torch.tensor(other, device=self.device, dtype=self.dtype)

    n_elements = self.numel()
    if n_elements == 0:
        return self

    if torch.broadcast_shapes(self.shape, other_t.shape) != self.shape:
        raise RuntimeError(
            "hypot_: the size of tensor other "
            f"{tuple(other_t.shape)} must be broadcastable to the size of "
            f"tensor self {tuple(self.shape)}"
        )

    if self.is_contiguous() and other_t.shape == self.shape and other_t.is_contiguous():
        _launch_hypot_inplace_flat(self, other_t)
        return self

    shapes = torch.tensor(
        list(reversed(self.shape)), dtype=torch.int64, device=self.device
    )
    x_strides = torch.tensor(
        list(reversed(self.stride())), dtype=torch.int64, device=self.device
    )
    y_strides = torch.tensor(
        list(reversed(_effective_broadcast_strides(other_t, self.shape))),
        dtype=torch.int64,
        device=self.device,
    )
    _launch_hypot_inplace_strided(self, other_t, shapes, x_strides, y_strides)
    return self


__all__ = ["hypot", "hypot_"]
