from typing import Generator

import pytest
import torch

from . import base, consts

# Exponent to raise each matrix to during benchmarking.
MATRIX_POWER_N = 5


class MatrixPowerBenchmark(base.BlasBenchmark):
    def set_more_shapes(self):
        # Square matrices of increasing size.
        return [(1, m, m, m) for m in (256, 512, 1024, 2048)]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            m = shape[-1]
            yield from self.input_fn(m, dtype, self.device)


def _input_fn(m, cur_dtype, device):
    inp = torch.randn([m, m], dtype=cur_dtype, device=device)
    yield inp, MATRIX_POWER_N


def _input_fn_out(m, cur_dtype, device):
    inp = torch.randn([m, m], dtype=cur_dtype, device=device)
    out = torch.empty([m, m], dtype=cur_dtype, device=device)
    yield inp, MATRIX_POWER_N, {"out": out}


@pytest.mark.matrix_power
def test_matrix_power():
    bench = MatrixPowerBenchmark(
        op_name="matrix_power",
        input_fn=_input_fn,
        torch_op=torch.matrix_power,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.matrix_power_out
def test_matrix_power_out():
    bench = MatrixPowerBenchmark(
        op_name="matrix_power_out",
        input_fn=_input_fn_out,
        torch_op=torch.matrix_power,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
