import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.lift_fresh_copy
def test_lift_fresh_copy():
    bench = utils.GenericBenchmark(
        input_fn=lambda shape, dtype, device: (
            iter([(torch.randn(shape, dtype=dtype, device=device),)])
        ),
        op_name="lift_fresh_copy",
        torch_op=torch.ops.aten.lift_fresh_copy,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
