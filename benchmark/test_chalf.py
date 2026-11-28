import pytest
import torch

from . import base, consts


@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "torch_gcu has no PrivateUse1 kernel for aten::chalf; its CPU fallback "
        "aborts the process, so there is no native baseline"
    ),
)
@pytest.mark.chalf
def test_chalf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="chalf",
        torch_op=torch.Tensor.chalf,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
