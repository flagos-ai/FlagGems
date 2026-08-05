import pytest
import torch

from . import base


def view_as_complex_input_fn(shape, dtype, device):
    # view_as_complex only supports float32 and float64.
    # Last dim must be 2 (real/imag parts).
    complex_shape = list(shape) + [2]
    # Use torch.randn directly: generate_tensor_input excludes float64
    # from FLOAT_DTYPES and would return None for it.
    inp = torch.randn(complex_shape, dtype=dtype, device=device)
    yield inp, {}


@pytest.mark.view_as_complex
def test_view_as_complex():
    bench = base.GenericBenchmark(
        input_fn=view_as_complex_input_fn,
        op_name="view_as_complex",
        torch_op=torch.view_as_complex,
        # view_as_complex only supports float32 and float64 per PyTorch docs
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
