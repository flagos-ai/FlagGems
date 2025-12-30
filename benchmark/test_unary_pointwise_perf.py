from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from benchmark.performance_utils import Benchmark, SkipVersion, generate_tensor_input

vendor_name = flag_gems.vendor_name
fp64_is_supported = flag_gems.runtime.device.support_fp64


class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


forward_operations = [
    ("abs", torch.abs, FLOAT_DTYPES),
    ("angle", torch.angle, COMPLEX_DTYPES + [torch.float32] + INT_DTYPES + BOOL_DTYPES),
    ("erf", torch.erf, FLOAT_DTYPES),
    ("exp", torch.exp, FLOAT_DTYPES),
    ("exp2", torch.exp2, FLOAT_DTYPES),
    ("neg", torch.neg, FLOAT_DTYPES),
    ("reciprocal", torch.reciprocal, FLOAT_DTYPES),
    ("sqrt", torch.sqrt, FLOAT_DTYPES),
    ("rsqrt", torch.rsqrt, FLOAT_DTYPES),
    ("logical_not", torch.logical_not, INT_DTYPES + BOOL_DTYPES),
    ("log", torch.log, FLOAT_DTYPES),
    # ("triu", torch.triu, FLOAT_DTYPES),  # do not support 1d shapes
    # Dropout
    ("dropout", torch.nn.Dropout(p=0.5), FLOAT_DTYPES),
    # Activation operations
    ("celu", torch.nn.functional.celu, FLOAT_DTYPES),
    ("elu", torch.nn.functional.elu, FLOAT_DTYPES),
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
    ("relu", torch.nn.functional.relu, FLOAT_DTYPES),
    ("softplus", torch.nn.functional.softplus, FLOAT_DTYPES),
    ("sigmoid", torch.sigmoid, FLOAT_DTYPES),
    ("log_sigmoid", torch.nn.functional.logsigmoid, FLOAT_DTYPES),
    ("silu", torch.nn.functional.silu, FLOAT_DTYPES),
    # Trigonometric operations
    ("cos", torch.cos, FLOAT_DTYPES),
    ("sin", torch.sin, FLOAT_DTYPES),
    ("tan", torch.tan, FLOAT_DTYPES),
    ("tanh", torch.tanh, FLOAT_DTYPES),
    ("atan", torch.atan, FLOAT_DTYPES),
    # Bitwise operations
    ("bitwise_not", torch.bitwise_not, INT_DTYPES),
    # Numerical Checks
    ("isinf", torch.isinf, FLOAT_DTYPES),
    ("isnan", torch.isnan, FLOAT_DTYPES),
    ("isfinite", torch.isfinite, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in forward_operations
    ],
)
def test_general_unary_pointwise_perf(op_name, torch_op, dtypes):
    if vendor_name == "kunlunxin":
        if op_name in ["celu"] and SkipVersion("torch", "<2.5"):
            pytest.skip(
                "There is an error in kunlunxin torch 2.0 aten, please use torch 2.5 instead"
            )
    bench = UnaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


forward_inplace_operations = [
    ("abs_", torch.abs_, FLOAT_DTYPES),
    # ("angle", torch.angle, COMPLEX_DTYPES + [torch.float32] + INT_DTYPES + BOOL_DTYPES),
    ("erf_", torch.erf_, FLOAT_DTYPES),
    ("exp_", torch.exp_, FLOAT_DTYPES),
    ("exp2_", torch.exp2_, FLOAT_DTYPES),
    ("neg_", torch.neg_, FLOAT_DTYPES),
    ("reciprocal_", torch.reciprocal_, FLOAT_DTYPES),
    ("sqrt_", torch.sqrt_, FLOAT_DTYPES),
    ("rsqrt_", torch.rsqrt_, FLOAT_DTYPES),
    # Activation operations
    ("celu_", torch.nn.functional.celu_, FLOAT_DTYPES),
    ("elu_", torch.nn.functional.elu_, FLOAT_DTYPES),
    ("gelu_", torch.ops.aten.gelu_.default, FLOAT_DTYPES),
    ("relu_", torch.relu_, FLOAT_DTYPES),
    ("sigmoid_", torch.sigmoid_, FLOAT_DTYPES),
    ("silu_", lambda a: torch.nn.functional.silu(a, inplace=True), FLOAT_DTYPES),
    # Trigonometric operations
    ("cos_", torch.cos_, FLOAT_DTYPES),
    ("sin_", torch.sin_, FLOAT_DTYPES),
    ("tan_", torch.tan_, FLOAT_DTYPES),
    ("tanh_", torch.tanh_, FLOAT_DTYPES),
    ("atan_", torch.atan_, FLOAT_DTYPES),
    # Bitwise operations
    ("bitwise_not_", lambda a: a.bitwise_not_(), INT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in forward_inplace_operations
    ],
)
def test_general_inplace_unary_pointwise_perf(op_name, torch_op, dtypes):
    if vendor_name == "kunlunxin":
        if op_name in ["celu_"] and SkipVersion("torch", "<2.5"):
            pytest.skip(
                "There is an error in kunlunxin torch 2.0 aten, please use torch 2.5 instead"
            )
    bench = UnaryPointwiseBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=dtypes, is_inplace=True
    )
    bench.run()


backward_operations = [
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name + "_backward", None),
        )
        for name, op, dtype in backward_operations
    ],
)
def test_general_unary_pointwise_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryPointwiseBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()


class ToCopyBenchmark(UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=torch.float32, device=self.device)
            yield inp, {"dtype": cur_dtype}


@pytest.mark.to_copy
def test_to_copy_perf():
    bench = ToCopyBenchmark(
        op_name="to_copy",
        torch_op=torch.ops.aten._to_copy,
        dtypes=[torch.float16, torch.bfloat16]
        + ([torch.float64] if fp64_is_supported else []),
    )
    bench.run()


class CopyInplaceBenchmark(Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            dst = generate_tensor_input(shape, cur_dtype, self.device)
            src = generate_tensor_input(shape, cur_dtype, self.device)
            yield dst, src


@pytest.mark.copy_
@pytest.mark.skipif(
    SkipVersion("torch", "<2.4"),
    reason="The copy operator implement required for torch >= 2.4",
)
def test_copy_inplace_perf():
    bench = CopyInplaceBenchmark(
        op_name="copy_",
        torch_op=torch.ops.aten.copy_,
        dtypes=FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


class EluBackwardBenchmark(UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            grad_out = torch.randn_like(inp)
            alpha = 1.0
            scale = 1.0
            input_scale = 1.0
            is_result = False

            yield grad_out, alpha, scale, input_scale, is_result, inp


@pytest.mark.elu
def test_elu_backward_perf():
    bench = EluBackwardBenchmark(
        op_name="elu_backward",
        torch_op=torch.ops.aten.elu_backward,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class GluBenchmark(UnaryPointwiseBenchmark):
    # Glu test requires even numbers
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(1, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(1, 15, 4)]
        return special_shapes_2d + sp_shapes_3d


@pytest.mark.glu
def test_glu_perf():
    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch.nn.functional.glu,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.glu
def test_glu_backward_perf():
    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch.nn.functional.glu,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()


class BinaryPointwiseBenchmark(Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            shift_amount = torch.randint(0, 8, shape, dtype=cur_dtype, device="cpu").to(
                self.device
            )
            yield inp1, shift_amount


@pytest.mark.bitwise_left_shift
def test_bitwise_left_shift_perf():
    bench = BinaryPointwiseBenchmark(
        op_name="bitwise_left_shift",
        torch_op=torch.bitwise_left_shift,
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_right_shift
def test_bitwise_right_shift_perf():
    bench = BinaryPointwiseBenchmark(
        op_name="bitwise_right_shift",
        torch_op=torch.bitwise_right_shift,
        dtypes=INT_DTYPES,
    )
    bench.run()
