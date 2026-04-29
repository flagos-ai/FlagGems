import pytest
import torch

import flag_gems

from . import base, consts, utils

vendor_name = flag_gems.vendor_name


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 3.14, 2.71


@pytest.mark.threshold
@pytest.mark.skipif(
    vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="kunlunxin torch aten 2.0 supports threshold but not for float16",
)
def test_threshold():
    bench = base.GenericBenchmark(
        op_name="threshold",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.threshold,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
