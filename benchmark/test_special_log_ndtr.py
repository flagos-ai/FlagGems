import pytest
import torch

from . import base


@pytest.mark.special_log_ndtr
def test_special_log_ndtr():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_log_ndtr",
        torch_op=torch.ops.aten.special_log_ndtr,
        # special_log_ndtr only supports float32 (aten has no Half/BFloat16 dispatch)
        dtypes=[torch.float32],
    )
    bench.run()
