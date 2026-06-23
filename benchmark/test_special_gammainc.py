import pytest
import torch

from . import base, utils


@pytest.mark.special_gammainc
def test_special_gammainc():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_gammainc",
        torch_op=torch.special.gammainc,
        # float32 only: gammainc series expansion is numerically unstable in lower precisions
        dtypes=[torch.float32],
    )
    bench.run()


def _input_fn_out(shape, dtype, device):
    x = utils.generate_tensor_input(shape, dtype, device)
    y = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty_like(x)
    yield x, y, {"out": out}


@pytest.mark.special_gammainc_out
def test_special_gammainc_out():
    bench = base.GenericBenchmark(
        op_name="special_gammainc_out",
        input_fn=_input_fn_out,
        torch_op=torch.ops.aten.special_gammainc.out,
        # float32 only: gammainc series expansion is numerically unstable in lower precisions
        dtypes=[torch.float32],
    )
    bench.run()
