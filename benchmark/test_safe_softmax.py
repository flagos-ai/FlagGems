from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class SafeSoftmaxBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            yield inp, -1, None


@pytest.mark.safe_softmax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_safe_softmax():
    bench = SafeSoftmaxBenchmark(
        op_name="safe_softmax",
        torch_op=torch.ops.aten._safe_softmax,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
