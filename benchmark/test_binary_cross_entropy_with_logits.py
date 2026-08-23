import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.rand(shape, dtype=dtype, device=device)
    yield inp, target


@pytest.mark.binary_cross_entropy_with_logits
def test_binary_cross_entropy_with_logits():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="binary_cross_entropy_with_logits",
        torch_op=torch.ops.aten.binary_cross_entropy_with_logits,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
