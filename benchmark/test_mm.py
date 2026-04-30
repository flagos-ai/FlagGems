import pytest
import torch

import flag_gems

from . import base, consts


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


@pytest.mark.mm
def test_mm():
    bench = base.BlasBenchmark(
        op_name="mm",
        input_fn=mm_input_fn,
        torch_op=torch.Tensor.mm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


class MmSelfTransposeBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        m, k = args[0].shape
        return 2 * m * m * k


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    inp = torch.randn([k, m], dtype=cur_dtype, device=device).t()

    yield inp,


def torch_mm_self_transpose(inp):
    return torch.mm(inp, inp.t())


@pytest.mark.mm
def test_mm_self_transpose_benchmark():
    bench = MmSelfTransposeBenchmark(
        op_name="mm_self_transpose",
        input_fn=_input_fn,
        torch_op=torch_mm_self_transpose,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


class Int8MmBenchmark(base.GenericBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def get_input_iter(self, dtype) -> base.Generator:
        for m, n, k in self.shapes:
            yield from self.input_fn(m, n, k, dtype, self.device)

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2

        return total_flops


def _int8_mm_input_fn(m, n, k, cur_dtype, device):
    inp1 = torch.randint(-128, 127, [m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randint(-128, 127, [k, n], dtype=cur_dtype, device=device)

    yield inp1, inp2


@pytest.mark.int8_mm
def test_int8_mm():
    bench = Int8MmBenchmark(
        op_name="int8_mm",
        input_fn=_int8_mm_input_fn,
        torch_op=torch._int_mm,
        dtypes=[torch.int8],
    )

    bench.set_gems(flag_gems.int8_mm)
    bench.run()


def _int8_mm_input_fn_self_transpose(shape, cur_dtype, device):
    m, k = shape
    inp = torch.randint(-128, 127, [k, m], dtype=cur_dtype, device=device).t()

    yield inp, inp.t()


@pytest.mark.int8_mm
def test_int8_mm_self_transpose_benchmark():
    bench = MmSelfTransposeBenchmark(
        op_name="int8_mm_self_transpose",
        input_fn=_int8_mm_input_fn_self_transpose,
        torch_op=torch._int_mm,
        dtypes=[torch.int8],
    )

    bench.set_gems(flag_gems.int8_mm)
    bench.run()
