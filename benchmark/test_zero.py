from typing import Generator

import pytest
import torch

from . import base, consts, utils


class ZeroBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            yield inp,


@pytest.mark.zero
def test_zero():
    bench = ZeroBenchmark(
        op_name="zero",
        torch_op=torch.ops.aten.zero,
        dtypes=consts.FLOAT_DTYPES,
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


def _input_fn_out(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield input, {"out": out}


@pytest.mark.zero_out
def test_zero_out():
    bench = base.GenericBenchmark(
        op_name="zero_out",
        input_fn=_input_fn_out,
        torch_op=torch.ops.aten.zero.out,
    )
    bench.run()
