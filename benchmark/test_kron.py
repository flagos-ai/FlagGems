import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as base
from . import utils


class KronBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    inp1 = base.generate_tensor_input(shape, dtype, device)
    inp2 = base.generate_tensor_input(shape, dtype, device)
    yield inp1, inp2


@pytest.mark.kron
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="only support torch >= 2.5.",
)
def test_kron():
    bench = KronBenchmark(
        op_name="kron",
        input_fn=_input_fn,
        torch_op=torch.kron,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
