import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.saturate_weight_to_fp16
def test_saturate_weight_to_fp16():
    # Note: PyTorch's _saturate_weight_to_fp16 has a broken CPU implementation
    # that crashes, so we benchmark against torch.clamp as the reference
    def reference_op(x):
        return torch.clamp(x, -65504.0, 65504.0)

    bench = base.UnaryPointwiseBenchmark(
        op_name="saturate_weight_to_fp16",
        torch_op=reference_op,
        gems_op=flag_gems._saturate_weight_to_fp16,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
