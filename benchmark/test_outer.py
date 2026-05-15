from typing import Generator

import pytest
import torch

from . import base, consts


class OuterBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, dtype, self.device)


def _input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.outer
def test_outer():
    bench = OuterBenchmark(
        op_name="outer",
        input_fn=_input_fn,
        torch_op=torch.Tensor.outer,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
