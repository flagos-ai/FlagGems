import pytest
import torch

from . import base


def alias_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {}


@pytest.mark.alias
def test_alias():
    bench = base.GenericBenchmark(
        input_fn=alias_input_fn,
        op_name="alias",
        torch_op=torch.ops.aten.alias.default,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
