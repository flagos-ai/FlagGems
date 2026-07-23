from typing import Generator

import pytest
import torch

from . import base, consts


class SpecialSoftmaxBenchmark(base.Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, -1, None


@pytest.mark.special_softmax
def test_special_softmax():
    bench = SpecialSoftmaxBenchmark(
        op_name="special_softmax",
        torch_op=torch.special.softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
