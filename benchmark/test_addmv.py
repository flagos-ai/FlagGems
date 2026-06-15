from typing import Generator

import pytest
import torch

from . import base, consts


class AddmvBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, dtype, self.device)


def _input_fn(m, n, cur_dtype, device):
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    bias = torch.randn([m], dtype=cur_dtype, device=device)

    # torch.addmv(bias, mat, vec)
    yield bias, mat, vec


@pytest.mark.addmv
def test_addmv():
    bench = AddmvBenchmark(
        op_name="addmv",
        input_fn=_input_fn,
        torch_op=torch.addmv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
