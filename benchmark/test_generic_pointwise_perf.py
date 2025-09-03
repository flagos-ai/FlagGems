import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES
from benchmark.conftest import BenchLevel, Config
from benchmark.performance_utils import (
    GenericBenchmark,
    GenericBenchmarkExcluse1D,
    generate_tensor_input,
    unary_input_fn,
    vendor_name,
)


def flip_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    if len(shape) > 1:
        yield inp, {"dims": (0, 1)}
    else:
        yield inp, {"dims": (0,)}


def where_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    condition = inp1 > 0
    yield condition, inp1, inp2


def nan_to_num_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    inp.view(-1)[0] = float("nan")
    if inp.numel() > 1:
        inp.view(-1)[1] = float("inf")
    if inp.numel() > 2:
        inp.view(-1)[2] = float("-inf")
    yield inp,


def clamp_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    inp3 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2, inp3
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # scalar or None situation
        yield inp1, inp2, None
        yield inp1, None, 3.14


def threshold_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 3.14, 2.71


def addcmul_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    inp3 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2, inp3, {"value": 0.5}


def addcdiv_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    inp3 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "nan_to_num",
            torch.nan_to_num,
            nan_to_num_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.nan_to_num,
        ),
        pytest.param(
            "clamp",
            torch.clamp,
            clamp_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.clamp,
        ),
        pytest.param(
            "flip",
            torch.flip,
            flip_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.flip,
        ),
        pytest.param(
            "where", torch.where, where_input_fn, FLOAT_DTYPES, marks=pytest.mark.where
        ),
        pytest.param(
            "threshold",
            torch.nn.functional.threshold,
            threshold_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.threshold,
        ),
        pytest.param(
            "addcmul",
            torch.addcmul,
            addcmul_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.addcmul,
        ),
        pytest.param(
            "addcdiv",
            torch.addcdiv,
            addcmul_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.addcdiv,
        ),
    ],
)
def test_generic_pointwise_benchmark(op_name, torch_op, input_fn, dtypes):
    if vendor_name == "kunlunxin":
        if op_name in ["threshold"]:
            pytest.skip("TODOFIX")
    bench = GenericBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "triu",
            torch.triu,
            unary_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.triu,
        ),
    ],
)
def test_generic_pointwise_benchmark_exclude_1d(op_name, torch_op, input_fn, dtypes):
    bench = GenericBenchmarkExcluse1D(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()
