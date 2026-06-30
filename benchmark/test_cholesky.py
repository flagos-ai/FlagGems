import pytest
import torch

from . import base


def cholesky_input_fn(shape, dtype, device):
    # Cholesky requires square matrices
    # Use smaller matrix sizes to avoid OOM
    n = min(shape[-1], 1024)  # Limit matrix size

    # Use smaller batch dimension if needed
    batch = shape[0] if len(shape) > 2 else 1
    if batch > 8:
        batch = 8

    shape = (batch, n, n)

    # Create a symmetric positive-definite matrix
    # Use a stable method: A = B @ B.T + alpha * I
    # where B has full rank and alpha > 0
    B = torch.randn(shape, dtype=dtype, device=device)
    A = B @ B.transpose(-2, -1)
    # Add a large enough diagonal to ensure positive definiteness
    A = A + torch.eye(n, dtype=dtype, device=device) * (n * 0.1)
    yield A,


@pytest.mark.cholesky
def test_cholesky():
    bench = base.GenericBenchmark2DOnly(
        input_fn=cholesky_input_fn,
        op_name="cholesky",
        torch_op=torch.cholesky,
        # float64 supported on NVIDIA via cuSOLVER; use consts.FLOAT_DTYPES equivalent
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
