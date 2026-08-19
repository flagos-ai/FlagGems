import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    # shape is (N,) — use the first element as window_length
    window_length = shape[0] if isinstance(shape, (list, tuple)) else shape
    yield {
        "window_length": window_length,
        "periodic": True,
        "dtype": dtype,
        "device": device,
    },


@pytest.mark.bartlett_window
def test_bartlett_window():
    bench = base.GenericBenchmark(
        op_name="bartlett_window",
        input_fn=_input_fn,
        torch_op=torch.bartlett_window,
        # bartlett_window only supports float32 (torch.bartlett_window default dtype)
        dtypes=[torch.float32],
    )
    bench.run()
