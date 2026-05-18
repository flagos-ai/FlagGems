import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    64,
    (512, 1, 1),
    32,
    False,
    False,
)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
    config=config_,
)
@triton.jit
def where_inner(condition, self, other):
    return tl.where(condition, self, other)


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS WHERE_SELF_OUT")
    result_type = torch.result_type(self, other)
    if out is not None:
        assert (
            out.dtype == result_type
        ), f"Expected out type to be {result_type}, but got {out.dtype}."

    c, a, b = list(
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            (condition, self, other),
        )
    )

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    devices = map(lambda x: x.device, (c, a, b))
    devices = list(filter(lambda k: k.type != "cpu", devices))

    assert len(devices), "CPU only. There seems a mistake to dispatch to here."

    device = devices[0]
    if c.device != device and c.ndim == 0:
        c = c.to(device)
    if a.device != device and a.ndim == 0:
        a = a.to(device)
    if b.device != device and b.ndim == 0:
        b = b.to(device)

    assert (
        len(set(devices)) == 1
    ), f"Expected all tensors to be on the same device, but found at least two devices, {devices}"
    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
    orig_out_shape = out_shape

    # expand 0D scalars to at least 1D, rank_0 kernel not supported on TPU
    # Use torch.full instead of expand().contiguous() because .contiguous()
    # on TPU for bool tensors is buggy (produces alternating True/False)
    if out_shape == ():
        out_shape = (1,)
    if c.ndim == 0:
        c = torch.full(out_shape, c.item(), dtype=torch.bool, device=device)
    if a.ndim == 0:
        a = torch.full(out_shape, a.item(), dtype=result_type, device=device)
    if b.ndim == 0:
        b = torch.full(out_shape, b.item(), dtype=result_type, device=device)

    ndim = len(out_shape)

    if ndim > 4:
        # dims where any tensor has size 1 but out_shape > 1 are protected
        protected = set()
        for t in (c, a, b):
            pad = ndim - t.ndim
            for i in range(t.ndim):
                if t.shape[i] == 1 and out_shape[pad + i] > 1:
                    protected.add(pad + i)

        # merge consecutive non-protected dims
        merged = []
        i = 0
        while i < ndim:
            if i in protected:
                merged.append(out_shape[i])
                i += 1
            else:
                prod = 1
                while i < ndim and i not in protected:
                    prod *= out_shape[i]
                    i += 1
                merged.append(prod)

        # still >4D after merge: merge leading dims
        if len(merged) > 4:
            extra = len(merged) - 4
            merged = [math.prod(merged[: extra + 1])] + merged[extra + 1 :]
        merged = tuple(merged)

        def _reshape(t):
            while t.ndim < ndim:
                t = t.unsqueeze(0)
            return t.reshape(merged)

        c = _reshape(c)
        a = _reshape(a)
        b = _reshape(b)
        if out is not None:
            out = out.reshape(merged)
        out_shape = merged

    if out is None:
        out = torch.empty(out_shape, dtype=result_type, device=device)
    elif out.shape != out_shape:
        out = out.reshape(out_shape)

    ndim = max(c.ndim, a.ndim, b.ndim)
    where_inner.instantiate(ndim)
    where_inner(c, a, b, out0=out)

    if out.shape != orig_out_shape:
        out = out.reshape(orig_out_shape)
    return out


def where_self(condition, self, other):
    logger.debug("GEMS WHERE_SELF")
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    logger.debug("GEMS WHERE_SCALAR_SELF")
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    logger.debug("GEMS WHERE_SCALAR_OTHER")
    return where_self_out(condition, self, other)