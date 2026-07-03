import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark


def isneginf_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    if inp.numel() > 0:
        inp.flatten()[0] = float("-inf")
    yield inp,


def isneginf_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    if inp.numel() > 0:
        inp.flatten()[0] = float("-inf")
    yield inp, {"out": torch.empty(shape, dtype=torch.bool, device=device)}


@pytest.mark.isneginf
def test_isneginf():
    bench = GenericBenchmark(
        input_fn=isneginf_input_fn,
        op_name="isneginf",
        torch_op=torch.isneginf,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.isneginf_out
def test_isneginf_out():
    bench = GenericBenchmark(
        input_fn=isneginf_out_input_fn,
        op_name="isneginf_out",
        torch_op=torch.isneginf,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
