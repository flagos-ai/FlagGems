from typing import Generator

import pytest
import torch

from . import base, consts


class VdotBenchmark(base.BlasBenchmark):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            m = shape[0]
            yield from self.input_fn(m, dtype, self.device)


@pytest.mark.vdot
def test_vdot():
    def vdot_input_fn(m, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = VdotBenchmark(
        input_fn=vdot_input_fn,
        op_name="vdot",
        torch_op=torch.Tensor.vdot,
        dtypes=consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
