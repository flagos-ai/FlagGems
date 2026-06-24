import pytest
import torch

from . import base


def _ndtri_input_fn(shape, dtype, device):
    # ndtri domain is (0, 1); sample probabilities away from the endpoints
    x = torch.rand(shape, dtype=dtype, device=device) * 0.998 + 0.001
    yield (x,)


def _ndtri_input_fn_out(shape, dtype, device):
    x = torch.rand(shape, dtype=dtype, device=device) * 0.998 + 0.001
    out = torch.empty_like(x)
    yield x, {"out": out}


@pytest.mark.special_ndtri
def test_special_ndtri():
    bench = base.GenericBenchmark(
        op_name="special_ndtri",
        input_fn=_ndtri_input_fn,
        torch_op=torch.ops.aten.special_ndtri,
        # torch reference ndtri only supports float32/float64 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_ndtri_out
def test_special_ndtri_out():
    bench = base.GenericBenchmark(
        op_name="special_ndtri_out",
        input_fn=_ndtri_input_fn_out,
        torch_op=torch.ops.aten.special_ndtri.out,
        # torch reference ndtri only supports float32/float64 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()
