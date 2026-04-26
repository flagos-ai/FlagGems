import pytest
import torch

import flag_gems

from . import attri_util as attrs
from . import performance_utils as base


def _input_fn(shape, cur_dtype, device):
    inp1 = base.generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 3.14, 2.71


@pytest.mark.threshold
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin" and base.SkipVersion("torch", "<2.5"),
    reason="kunlunxin torch aten 2.0 supports threshold but not for float16",
)
def test_threshold():
    bench = base.GenericBenchmark(
        op_name="threshold",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.threshold,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
