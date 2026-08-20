import pytest
import torch

from . import base, consts, utils


def _aminmax_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    # ``_aminmax`` reduces the whole tensor; there is no dim argument.
    yield inp,


def _aminmax_out_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    min_out = torch.empty((), dtype=cur_dtype, device=device)
    max_out = torch.empty((), dtype=cur_dtype, device=device)
    # The aten ``.out`` overload takes keyword-only ``out0``/``out1``.
    yield inp, {"out0": min_out, "out1": max_out}


class AminmaxAllReduceBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from _aminmax_input_fn(shape, cur_dtype, self.device)


class AminmaxOutBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from _aminmax_out_input_fn(shape, cur_dtype, self.device)


@pytest.mark.underscore_aminmax
def test__aminmax():
    bench = AminmaxAllReduceBenchmark(
        op_name="_aminmax",
        torch_op=torch._aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_aminmax_out
def test__aminmax_out():
    bench = AminmaxOutBenchmark(
        op_name="_aminmax_out",
        torch_op=torch.ops.aten._aminmax.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
