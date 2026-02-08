"""
Performance benchmark tests for tril operator
"""
import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmarkExcluse1D


def tril_input_fn(shape, cur_dtype, device):
    """Generate input for tril with diagonal parameter"""
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    # Test with diagonal=0 (default)
    yield inp, {"diagonal": 0}


@pytest.mark.tril
def test_tril_perf():
    """Performance test for tril operator"""
    bench = GenericBenchmarkExcluse1D(
        input_fn=tril_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril_
@pytest.mark.inplace
def test_tril_inplace_perf():
    """Performance test for tril_ (in-place) operator"""

    def tril_input_fn_inplace(shape, cur_dtype, device):
        inp = torch.randn(shape, dtype=cur_dtype, device=device)
        yield inp, {"diagonal": 0}

    bench = GenericBenchmarkExcluse1D(
        input_fn=tril_input_fn_inplace,
        op_name="tril_",
        torch_op=lambda x, diagonal=0: x.tril_(diagonal),
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
