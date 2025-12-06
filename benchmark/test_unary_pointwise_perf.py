import gc
import logging
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
    BenchmarkMetrics,
    BenchmarkResult,
)
from benchmark.performance_utils import (
    Benchmark,
    Config,
    generate_tensor_input,
    vendor_name,
)

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

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


@pytest.mark.elu_backward
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


@pytest.mark.glu_backward
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
            shift_amount = torch.randint(
                0, 8, shape, dtype=cur_dtype, device=self.device
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


class TEBenchmarkResult(BenchmarkResult):
    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode},"
            f"level={self.level})\n"
        )
        col_names = [
            f"{'Status':<12}",
            f"{'TE Latency (ms)':>20}",
            f"{'Gems Latency (ms)':>20}",
            f"{'Gems Speedup':>20}",
            f"{'TE GBPS':>20}",
            f"{'Gems GBPS':>20}",
            "          Size Detail",
        ]

        header_col_names = "".join(col_names)
        header_break = "\n" + "-" * (len(header_col_names) + 10)
        header = header_title + header_col_names + header_break

        metrics_lines = "".join(self._format_metrics(ele) for ele in self.result)
        return header + metrics_lines

    def _format_metrics(self, metrics: BenchmarkMetrics) -> str:
        status = "SUCCESS" if metrics.error_msg is None else "FAILED"
        latency_base_str = (
            f"{metrics.latency_base:.6f}" if metrics.latency_base is not None else "N/A"
        )
        latency_str = f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        speedup_str = f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"
        gbps_base_str = (
            f"{metrics.gbps_base:.3f}" if metrics.gbps_base is not None else "N/A"
        )
        gbps_str = f"{metrics.gbps:.3f}" if metrics.gbps is not None else "N/A"
        shape_detail_str = f"{metrics.shape_detail}"

        data_line = (
            f"\n{status:<12}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
            f"{gbps_base_str:>20}"
            f"{gbps_str:>20}"
            f"          {shape_detail_str}"
        )
        return data_line


class DregluBenchmark(Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = DEFAULT_METRICS + ["gbps"]
        self.to_bench_metrics = self.metrics

    def set_shapes(self, shape_file_path: str = None):
        core_shapes = [
            (1024 * 1024,),
            (4096, 2048),
            (16, 1024, 4096),
            (8, 512, 8192),
            (4, 128, 8, 2048),
        ]
        self.shapes = core_shapes

        if Config.bench_level.value == "comprehensive":
            additional_shapes = self.set_more_shapes()
            if additional_shapes:
                self.shapes = list(dict.fromkeys(self.shapes + additional_shapes))

    def set_more_shapes(self):
        special_shapes_2d = [(4096, 2**i) for i in range(8, 14, 2)]
        sp_shapes_3d = [(16, 1024, 2**i) for i in range(10, 15, 2)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype: torch.dtype) -> Generator:
        for input_shape in self.shapes:
            if input_shape[-1] % 2 != 0:
                continue
            input_tensor = generate_tensor_input(input_shape, cur_dtype, self.device)
            grad_output_shape = list(input_shape)
            grad_output_shape[-1] //= 2
            grad_output = generate_tensor_input(
                tuple(grad_output_shape), cur_dtype, self.device
            )
            yield grad_output, input_tensor

    def get_gbps(self, args: tuple, latency: float) -> float:
        if not latency or latency == 0:
            return 0.0
        grad_output, input_tensor = args
        element_size = grad_output.element_size()
        total_bytes = (grad_output.numel() + 2 * input_tensor.numel()) * element_size
        return total_bytes / (latency * 1e6)

    def run(self):
        if Config.query:
            super().run()
            return

        self.init_user_config()
        if "gbps" not in self.to_bench_metrics and any(
            m in self.to_bench_metrics for m in ["latency", "latency_base"]
        ):
            self.to_bench_metrics.append("gbps")

        for dtype in self.to_bench_dtypes:
            metrics_list = []
            for input_data in self.get_input_iter(dtype):
                metric = BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(input_data)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)
                    if "latency_base" in self.to_bench_metrics:
                        metric.latency_base = self.get_latency(
                            self.torch_op, *args, **kwargs
                        )
                    if "latency" in self.to_bench_metrics:
                        if not self.gems_op:
                            raise ValueError(
                                "GEMS operator not set. Use bench.set_gems()."
                            )
                        metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
                    if (
                        "speedup" in self.to_bench_metrics
                        and metric.latency is not None
                        and metric.latency > 0
                    ):
                        metric.speedup = metric.latency_base / metric.latency
                    if "gbps" in self.to_bench_metrics:
                        metric.gbps_base = self.get_gbps(
                            args, latency=metric.latency_base
                        )
                        metric.gbps = self.get_gbps(args, latency=metric.latency)
                except Exception as e:
                    metric.error_msg = str(e)
                    print(f"\nBenchmark failed for shape {metric.shape_detail}: {e}")
                finally:
                    metrics_list.append(metric)
                    gc.collect()

            if not metrics_list:
                continue

            result_formatter = TEBenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics_list,
            )
            print(result_formatter)
            logging.info(result_formatter.to_json())


@pytest.mark.skipif(
    not TE_AVAILABLE,
    reason="Transformer Engine backend is not available for reference.",
)
@pytest.mark.dreglu
def test_dreglu_perf():
    bench = DregluBenchmark(
        op_name="dreglu_backward",
        torch_op=lambda grad, inp: tex.dreglu(grad, inp, None),
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.dreglu)
    bench.run()


class RegluBenchmark(Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = DEFAULT_METRICS + ["gbps"]
        self.to_bench_metrics = self.metrics

    def set_shapes(self, shape_file_path: str = None):
        core_shapes = [
            (4096, 2048),
            (16, 1024, 4096),
            (8, 512, 8192),
            (4, 128, 8, 2048),
        ]
        self.shapes = core_shapes

        if Config.bench_level.value == "comprehensive":
            additional_shapes = self.set_more_shapes()
            if additional_shapes:
                self.shapes = list(dict.fromkeys(self.shapes + additional_shapes))

    def set_more_shapes(self):
        special_shapes_2d = [(4096, 2**i) for i in range(8, 14, 2)]
        sp_shapes_3d = [(16, 1024, 2**i) for i in range(10, 15, 2)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype: torch.dtype) -> Generator:
        for input_shape in self.shapes:
            if input_shape[-1] % 2 != 0:
                continue
            input_tensor = generate_tensor_input(input_shape, cur_dtype, self.device)
            yield input_tensor,

    def get_gbps(self, args: tuple, latency: float) -> float:
        if not latency or latency == 0:
            return 0.0
        (input_tensor,) = args
        element_size = input_tensor.element_size()
        total_bytes = (input_tensor.numel() + 0.5 * input_tensor.numel()) * element_size
        return total_bytes / (latency * 1e6)

    def run(self):
        if Config.query:
            super().run()
            return

        self.init_user_config()
        if "gbps" not in self.to_bench_metrics and any(
            m in self.to_bench_metrics for m in ["latency", "latency_base"]
        ):
            self.to_bench_metrics.append("gbps")

        for dtype in self.to_bench_dtypes:
            metrics_list = []
            for input_data in self.get_input_iter(dtype):
                metric = BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(input_data)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)
                    if "latency_base" in self.to_bench_metrics:
                        metric.latency_base = self.get_latency(
                            self.torch_op, *args, **kwargs
                        )
                    if "latency" in self.to_bench_metrics:
                        if not self.gems_op:
                            raise ValueError(
                                "GEMS operator not set. Use bench.set_gems()."
                            )
                        metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
                    if (
                        "speedup" in self.to_bench_metrics
                        and metric.latency is not None
                        and metric.latency > 0
                    ):
                        metric.speedup = metric.latency_base / metric.latency
                    if "gbps" in self.to_bench_metrics:
                        metric.gbps_base = self.get_gbps(
                            args, latency=metric.latency_base
                        )
                        metric.gbps = self.get_gbps(args, latency=metric.latency)
                except Exception as e:
                    metric.error_msg = str(e)
                    print(f"\nBenchmark failed for shape {metric.shape_detail}: {e}")
                finally:
                    metrics_list.append(metric)
                    gc.collect()

            if not metrics_list:
                continue

            result_formatter = TEBenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics_list,
            )
            print(result_formatter)
            logging.info(result_formatter.to_json())


@pytest.mark.skipif(
    not TE_AVAILABLE,
    reason="Transformer Engine backend is not available for reference.",
)
@pytest.mark.reglu
def test_reglu_perf():
    bench = RegluBenchmark(
        op_name="reglu_forward",
        torch_op=lambda inp: tex.reglu(inp, None),
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.reglu)
    bench.run()
