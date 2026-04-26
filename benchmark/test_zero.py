from typing import Generator

import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


class ZeroBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, dtype, self.device)
            yield inp,


@pytest.mark.zero
def test_zero():
    bench = ZeroBenchmark(
        op_name="zero",
        torch_op=torch.ops.aten.zero,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input,


@pytest.mark.zero_
def test_zero_inplace():
    bench = base.GenericBenchmark(
        op_name="zero_",
        input_fn=_input_fn,
        torch_op=torch.zero_,
    )
    bench.run()
