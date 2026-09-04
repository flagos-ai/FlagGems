import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    if isinstance(shape, int):
        n = m = shape
    elif len(shape) == 1:
        n = m = shape[0]
    else:
        n, m = shape[0], shape[1]

    # trace_backward materializes an (n, m) matrix; skip shapes whose element
    # count is too large to allocate for a benchmark.
    if n * m > 64 * 1024 * 1024:
        return

    grad = torch.randn((), dtype=dtype, device=device)
    yield grad, [n, m]


@pytest.mark.trace_backward
def test_trace_backward():
    bench = base.GenericBenchmark(
        op_name="trace_backward",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.trace_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
