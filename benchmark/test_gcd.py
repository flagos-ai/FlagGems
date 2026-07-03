import pytest
import torch

from .attri_util import INT_DTYPES
from .performance_utils import GenericBenchmark, binary_input_fn


def gcd_out_input_fn(shape, dtype, device):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(device)
    yield inp1, inp2, {"out": torch.empty(shape, dtype=dtype, device=device)}


@pytest.mark.gcd
def test_gcd():
    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="gcd",
        torch_op=torch.gcd,
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.gcd_out
def test_gcd_out():
    bench = GenericBenchmark(
        input_fn=gcd_out_input_fn,
        op_name="gcd_out",
        torch_op=torch.gcd,
        dtypes=INT_DTYPES,
    )
    bench.run()
