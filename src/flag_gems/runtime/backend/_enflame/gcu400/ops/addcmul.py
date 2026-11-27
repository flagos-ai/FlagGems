"""GCU400 addcmul: uses the shared 1D flat kernel when shape and dtype match and all contiguous.

The flat kernel and its tile tiers live in ``addcmul__common.py`` (shared with addcmul_;
that module also holds the implementation). Other cases (broadcast, non-contiguous, mixed
dtype, non-scalar value) keep the old ``addcmul_forward`` codegen fallback path.
"""

"""t8 4/8 addcmul candidate v1: hand-written 1D kernel for small shapes, common codegen otherwise.

Production uses pointwise_dynamic (gcu400 config max_tile=65536 / max_grid=(48,1,1)), where
small-shape latency is dominated by the codegen Python wrapper. The fast path requires three inputs
with same shape / same dtype / contiguous / scalar value / n<=65536; everything else falls back,
guaranteeing unchanged semantics. HP=True uses fp32 intermediates, bit-identical to production.
"""

import logging  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from ..utils.pointwise_dynamic import pointwise_dynamic  # noqa: E402

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcmul_forward(x, t1, t2, value):
    return x + value * t1 * t2


@triton.jit
def _addcmul_flat(
    X, T1, T2, Y, N, VALUE: tl.constexpr, BLOCK: tl.constexpr, HP: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0)
    t1 = tl.load(T1 + off, mask=m, other=0)
    t2 = tl.load(T2 + off, mask=m, other=0)
    if HP:
        y = (x.to(tl.float32) + VALUE * t1.to(tl.float32) * t2.to(tl.float32)).to(
            Y.dtype.element_ty
        )
    else:
        y = x + VALUE * t1 * t2
    tl.store(Y + off, y, mask=m)


_FAST_MAX = 65536
_BLOCK = 4096


def addcmul(inp, tensor1, tensor2, *, value=1.0, out=None):
    logger.debug("GEMS_ENFLAME ADDCMUL")
    logger.debug("GEMS_ENFLAME ADDCMUL_OUT")
    # Must import inside the function: pointwise_dynamic's codegen copies top-level
    # ``from .xxx import`` statements verbatim into generated code, where a relative import resolves to
    # ``gcu400.ops.addcmul.addcmul__common`` and raises ImportError.
    from .addcmul__common import try_addcmul_flat

    if out is None:
        flat_out = try_addcmul_flat(inp, tensor1, tensor2, value)
        if flat_out is not None:
            return flat_out
        return addcmul_forward(inp, tensor1, tensor2, value)
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    if list(out.shape) != list(broadcast_shape):
        out.resize_(broadcast_shape)
    flat_out = try_addcmul_flat(inp, tensor1, tensor2, value, out)
    if flat_out is not None:
        return flat_out
    addcmul_forward(inp, tensor1, tensor2, value, out0=out)
    return out


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out):
    return addcmul(inp, tensor1, tensor2, value=value, out=out)
