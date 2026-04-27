import itertools
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple

import torch

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]
BOOL_DTYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex64]

DEFAULT_WARMUP_COUNT = 1000
DEFAULT_ITER_COUNT = 100

# LEGACY_SHAPES are maintained for legacy benchmark SIZE settings and may be removed in the future.
# Do not reference this elsewhere.
LEGACY_SHAPES = [i * 64 for i in range(1, 22, 5)]
LEGACY_NON_BLAS_SHAPES = [(1024, shape) for shape in LEGACY_SHAPES]
LEGACY_BLAS_SHAPES = [(16, shape, shape, shape) for shape in LEGACY_SHAPES]

# Default shapes settings
DEFAULT_SHAPES = [
    (1024 * 1024 * 1024,),  # from perf
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),  # from perf
]


def model_shapes():
    # batch sizes * seq lengths
    BS = [1, 2, 3, 4, 8, 98, 256, 8192]
    # attn: wqkv, wo; ffn: w13, w2
    NK = [
        # extract from llama3-8b
        (1024, 4096),
        (128256, 4096),
        (14336, 4096),
        (4096, 14336),
        (4096, 4096),
        (6144, 4096),
        (28672, 4096),
        # extract from qwen2.5-7b
        (3584, 3584),
        (18944, 3584),
        (3584, 18944),
        (152064, 3584),
        (37888, 3584),
        (512, 3584),
        (4608, 3584),
    ]

    return [(4, bs, n, k) for bs, (n, k) in itertools.product(BS, NK)]


@dataclass
class BenchmarkMetrics:
    # Legacy shape information for backward compatibility
    # This field corresponds to the 'size' field in the previous version's benchmark.
    legacy_shape: Optional[int] = None
    # Detailed size info
    shape_detail: Optional[Tuple[int, ...]] = None
    # Latency base in ms
    latency_base: Optional[float] = None
    # Latency in ms
    latency: Optional[float] = None
    gbps_base: Optional[float] = None
    gbps: Optional[float] = None
    # Speedup over baseline
    speedup: Optional[float] = None
    # Accuracy over baseline (not implemented yet)
    accuracy: Optional[float] = None
    # TFLOPS (not implemented yet)
    tflops: Optional[float] = None
    # Utilization (not implemented yet)
    utilization: Optional[float] = None
    # Speedup compared to base data
    compared_speedup: Optional[float] = None
    # Error message
    error_msg: Optional[str] = None


ALL_AVAILABLE_METRICS = set(map(lambda x: x.name, fields(BenchmarkMetrics))) - {
    "legacy_shape",
    "shape_detail",
}

DEFAULT_METRICS = [
    metric
    for metric in ["latency_base", "latency", "speedup"]
    if metric in ALL_AVAILABLE_METRICS
]


def check_metric_dependencies(
    requested_metrics: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Checks if the requested metrics satisfy their dependencies.
    Returns True if the dependencies are satisfied, otherwise False.
    """
    # Predefined dependencies between metrics
    buildin_dependencies = {
        "speedup": ["latency", "latency_base"],
        "utilization": ["latency", "tflops"],
    }
    unsatisfied_metrics = []
    if requested_metrics is None:
        return unsatisfied_metrics

    satisfied_metrics = set()
    for metric in requested_metrics:
        if metric not in buildin_dependencies:
            # If the metric has no dependencies, it's automatically satisfied
            satisfied_metrics.add(metric)
        else:
            required_metrics = buildin_dependencies[metric]
            # Check if all dependencies are in the satisfied metrics list
            if not all(req in satisfied_metrics for req in required_metrics):
                unsatisfied_metrics.append(metric)
            else:
                satisfied_metrics.add(metric)
    return unsatisfied_metrics


def get_recommended_shapes(
    op_name: str, op_specified_shapes: Optional[List[Tuple[int, ...]]]
):
    def _shapes_sort(shapes):
        shapes = [shape if isinstance(shape, tuple) else (shape,) for shape in shapes]
        return sorted(shapes, key=lambda x: torch.tensor(x).prod().item())

    if op_specified_shapes:
        # TODO: handle situation that list as the basic element in shape.
        return _shapes_sort(op_specified_shapes)
    return _shapes_sort(DEFAULT_SHAPES)


class BenchMode(Enum):
    KERNEL = "kernel"
    OPERATOR = "operator"
    WRAPPER = "wrapper"


class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"


@dataclass
class OperationAttribute:
    op_name: str
    # Recommended core benchmark shapes for the given operation
    recommended_core_shapes: List[Tuple[int, ...]]
    shape_desc: str

    def __str__(self) -> str:
        return (
            f"{'Operator name':<40} |  {self.op_name}\n"
            f"{'Recommended Core Shapes[' + self.shape_desc + ']':<40} |  {self.recommended_core_shapes}\n"
        )

    def to_dict(self) -> dict:
        return self.__dict__


def custom_json_encoder(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class BenchmarkResult:
    """Record the benchmark result for each operator."""

    # Unique name of the operator
    op_name: str
    dtype: str
    mode: str
    level: str
    # Benchmark results
    result: List[BenchmarkMetrics]

    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode},"
            f"level={self.level})\n"
        )
        col_names = [
            f"{'Status':<10}",
            f"{'Torch Latency (ms)':>20}",
            f"{'Gems Latency (ms)':>20}",
            f"{'Gems Speedup':>20}",
        ]
        if self.result[0].tflops and self.result[0].tflops != 0.0:
            col_names.append(f"{'TFLOPS':>20}")
        if self.result[0].gbps is not None:
            col_names.append(f"{'Torch GBPS ':>20}")
            col_names.append(f"{'Gems GBPS ':>20}")
        col_names.append(f"{'Size Detail':>20}\n")
        header_col_names = " ".join(col_names)
        header_break = "-" * len(header_col_names) + "\n"
        header = header_title + header_col_names + header_break

        metrics_lines = "".join(self._format_metrics(ele) for ele in self.result)
        return header + metrics_lines

    def _format_metrics(self, metrics: BenchmarkMetrics) -> str:
        # self.gen_legacy_shape(metrics)
        # legacy_shape_str = (
        #     metrics.legacy_shape if metrics.legacy_shape is not None else "N/A"
        # )
        latency_base_str = (
            f"{metrics.latency_base:.6f}" if metrics.latency_base is not None else "N/A"
        )
        latency_str = f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        speedup_str = f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"
        torch_gbps_str = (
            f"{metrics.gbps_base:.3f}" if metrics.gbps_base is not None else "N/A"
        )
        gems_gbps_str = f"{metrics.gbps:.3f}" if metrics.gbps is not None else "N/A"
        if metrics.tflops and metrics.tflops != 0.0:
            tflops_str = (
                f"{metrics.tflops:.3f}" if metrics.tflops is not None else "N/A"
            )
        shape_detail_str = (
            metrics.shape_detail if metrics.shape_detail is not None else "N/A"
        )
        status = "SUCCESS" if metrics.error_msg is None else "FAILED"
        data_line = (
            f"{status:<10}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
        )
        if metrics.tflops and metrics.tflops != 0.0:
            data_line += f"{tflops_str:>20}"
        if metrics.gbps is not None:
            data_line += f"{torch_gbps_str:>20}{gems_gbps_str:>20}"
        data_line += " " * 10
        data_line += f"{shape_detail_str}\n"
        return data_line

    def gen_legacy_shape(self, metrics: BenchmarkMetrics) -> Optional[int]:
        first_shape = (
            metrics.shape_detail[0] if isinstance(metrics.shape_detail, list) else None
        )
        to_record_shape = (
            tuple(first_shape) if isinstance(first_shape, torch.Size) else None
        )

        if to_record_shape in LEGACY_NON_BLAS_SHAPES:
            metrics.legacy_shape = to_record_shape[-1]
        elif (
            isinstance(to_record_shape, tuple)
            and len(to_record_shape) == 2
            and to_record_shape[0] == 1024
        ):
            metrics.legacy_shape = to_record_shape[-1]
        else:
            metrics.legacy_shape = None

    def to_json(self) -> str:
        import json

        # Convert to dict and handle tuple serialization for shape_detail
        result_dict = asdict(self)
        return json.dumps(result_dict, default=custom_json_encoder)

    def to_dict(self) -> dict:
        return self.__dict__
