import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES
from benchmark.conftest import BenchLevel, Config
from benchmark.performance_utils import (
    GenericBenchmark,
    GenericBenchmarkExcluse1D,
    SkipVersion,
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


def clamp_min_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # scalar situation
        yield inp1, 3.14


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
            "clamp_min",
            torch.clamp_min,
            clamp_min_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.clamp_min,
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
    if vendor_name == "kunlunxin" and SkipVersion("torch", "<2.5"):
        if op_name in ["threshold"]:
            pytest.skip(
                "kunlunxin torch aten 2.0 supports threshold but not for float16"
            )
    bench = GenericBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


@pytest.mark.clamp_
def test_clamp_inplace():
    bench = GenericBenchmark(
        input_fn=clamp_input_fn,
        op_name="clamp_",
        torch_op=torch.clamp_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.clamp_min_
def test_clamp_min_inplace():
    bench = GenericBenchmark(
        input_fn=clamp_min_input_fn,
        op_name="clamp_min_",
        torch_op=torch.clamp_min_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.tril
def test_tril():
    bench = GenericBenchmarkExcluse1D(
        input_fn=unary_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.triu
def test_triu():
    bench = GenericBenchmarkExcluse1D(
        input_fn=unary_input_fn,
        op_name="triu",
        torch_op=torch.triu,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.triu_
def test_triu_inplace():
    bench = GenericBenchmarkExcluse1D(
        input_fn=unary_input_fn,
        op_name="triu_",
        torch_op=torch.Tensor.triu_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
