import pytest
import torch

from . import base, consts


# Helper function for broadcast_to benchmark
def _broadcast_to_op(x):
    # Broadcast to a larger but reasonable shape
    # For small 2D shapes (both dims <= 16), do 2x expansion
    # Otherwise, return as-is (no-op)
    if x.ndim == 2 and x.shape[0] <= 16 and x.shape[1] <= 16:
        shape = (x.shape[0] * 2, x.shape[1] * 2)
        return torch.broadcast_to(x, shape)
    elif x.ndim == 1 and x.shape[0] <= 64:
        # Broadcast 1D small shape to 2D
        return torch.broadcast_to(x, (2, x.shape[0]))
    else:
        # No-op for large shapes
        return x


@pytest.mark.broadcast_to
def test_broadcast_to():
    bench = base.UnaryPointwiseBenchmark(
        op_name="broadcast_to",
        torch_op=_broadcast_to_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
