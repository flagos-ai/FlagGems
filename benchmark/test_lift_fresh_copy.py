import pytest
import torch

from . import base, consts


@pytest.mark.lift_fresh_copy
def test_lift_fresh_copy():
    bench = base.GenericBenchmark(
        input_fn=lambda shape, dtype, device: (
            iter([(torch.randn(shape, dtype=dtype, device=device),)])
        ),
        op_name="lift_fresh_copy",
        torch_op=torch.ops.aten.lift_fresh_copy,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
