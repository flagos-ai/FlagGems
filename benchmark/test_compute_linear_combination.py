from typing import Generator

import pytest
import torch

from . import base, consts

# (m, n, k): coefficients is (m, n), input is (n, k), output is (m, k).
MNK_SHAPES = [
    (64, 64, 64),
    (128, 256, 512),
    (256, 512, 1024),
    (8, 1024, 4096),
]


class ComputeLinearCombinationBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return MNK_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) != 3:
                continue
            m, n, k = shape
            yield from self.input_fn(m, n, k, dtype, self.device)


def _input_fn(m, n, k, cur_dtype, device):
    # input is (n, k); coefficients is (m, n); output is (m, k)
    inp = torch.randn([n, k], dtype=cur_dtype, device=device)
    coeffs = torch.randn([m, n], dtype=cur_dtype, device=device)
    yield inp, coeffs


def _input_fn_out(m, n, k, cur_dtype, device):
    inp = torch.randn([n, k], dtype=cur_dtype, device=device)
    coeffs = torch.randn([m, n], dtype=cur_dtype, device=device)
    out = torch.empty([m, k], dtype=cur_dtype, device=device)
    yield inp, coeffs, {"out": out}


@pytest.mark.compute_linear_combination
def test_compute_linear_combination():
    bench = ComputeLinearCombinationBenchmark(
        op_name="compute_linear_combination",
        input_fn=_input_fn,
        torch_op=torch._compute_linear_combination,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.compute_linear_combination_out
def test_compute_linear_combination_out():
    bench = ComputeLinearCombinationBenchmark(
        op_name="compute_linear_combination_out",
        input_fn=_input_fn_out,
        torch_op=torch.ops.aten._compute_linear_combination.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
