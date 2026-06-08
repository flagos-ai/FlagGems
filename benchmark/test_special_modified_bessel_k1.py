import pytest
import torch

import flag_gems

from . import base


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="NVIDIA-only CUDA JIT kernel; not supported on other backends",
)
@pytest.mark.special_modified_bessel_k1
def test_special_modified_bessel_k1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_modified_bessel_k1",
        torch_op=torch.ops.aten.special_modified_bessel_k1,
        # PyTorch reference only supports float32 for this operator
        dtypes=[torch.float32],
    )
    bench.run()
