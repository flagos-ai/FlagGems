import json
import logging
import os

import pytest
import torch

import flag_gems
from benchmark.attri_util import (
    ALL_AVAILABLE_METRICS,
    BOOL_DTYPES,
    DEFAULT_ITER_COUNT,
    DEFAULT_WARMUP_COUNT,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchMode,
    OperationAttribute,
    get_recommended_shapes,
)
from flag_gems.runtime import torch_device_fn

device = flag_gems.device
vendor_name = flag_gems.vendor_name


class BenchConfig:
    def __init__(self):
        self.mode = BenchMode.KERNEL
        self.bench_level = BenchLevel.COMPREHENSIVE
        self.warm_up = DEFAULT_WARMUP_COUNT
        self.repetition = DEFAULT_ITER_COUNT
        if (
            vendor_name == "kunlunxin"
        ):  # Speed Up Benchmark Test, Big Shape Will Cause Timeout
            self.warm_up = 1
            self.repetition = 1
        self.record_log = False
        self.user_desired_dtypes = None
        self.user_desired_metrics = None
        self.shape_file = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
        self.query = False


Config = BenchConfig()


def pytest_addoption(parser):
    parser.addoption(
        (
            "--mode" if vendor_name != "kunlunxin" else "--fg_mode"
        ),  # TODO: fix pytest-* common --mode args
        action="store",
        default="kernel",
        required=False,
        choices=["kernel", "operator", "wrapper"],
        help=(
            "Specify how to measure latency, 'kernel' for device kernel, ",
            "'operator' for end2end operator or 'wrapper' for runtime wrapper.",
        ),
    )

    parser.addoption(
        "--level",
        action="store",
        default="comprehensive",
        required=False,
        choices=[level.value for level in BenchLevel],
        help="Specify the benchmark level: comprehensive, or core.",
    )

    parser.addoption(
        "--warmup",
        default=DEFAULT_WARMUP_COUNT,
        help="Number of warmup runs before benchmark run.",
    )

    parser.addoption(
        "--iter",
        default=DEFAULT_ITER_COUNT,
        help="Number of reps for each benchmark run.",
    )

    parser.addoption(
        "--query", action="store_true", default=False, help="Enable query mode"
    )

    parser.addoption(
        "--metrics",
        action="append",
        default=None,
        required=False,
        choices=ALL_AVAILABLE_METRICS,
        help=(
            "Specify the metrics we want to benchmark. "
            "If not specified, the metric items will vary according to the specified operation's category and name."
        ),
    )

    parser.addoption(
        "--dtypes",
        action="append",
        default=None,
        required=False,
        choices=[
            str(ele).split(".")[-1]
            for ele in FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES + [torch.cfloat]
        ],
        help=(
            "Specify the data types for benchmarks. "
            "If not specified, the dtype items will vary according to the specified operation's category and name."
        ),
    )

    parser.addoption(
        "--shape_file",
        action="store",
        default=os.path.join(os.path.dirname(__file__), "core_shapes.yaml"),
        required=False,
        help="Specify the shape file name for benchmarks. If not specified, a default shape list will be used.",
    )

    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="Benchmark info recorded in log files or not",
    )


def pytest_configure(config):
    global Config  # noqa: F824
    mode_value = config.getoption(
        "--mode" if vendor_name != "kunlunxin" else "--fg_mode"
    )
    Config.mode = BenchMode(mode_value)

    Config.query = config.getoption("--query")

    level_value = config.getoption("--level")
    Config.bench_level = BenchLevel(level_value)

    warmup_value = config.getoption("--warmup")
    Config.warm_up = int(warmup_value)

    iter_value = config.getoption("--iter")
    Config.repetition = int(iter_value)

    types_str = config.getoption("--dtypes")
    dtypes = [getattr(torch, dtype) for dtype in types_str] if types_str else types_str
    Config.user_desired_dtypes = dtypes

    metrics = config.getoption("--metrics")
    Config.user_desired_metrics = metrics

    shape_file_str = config.getoption("--shape_file")
    Config.shape_file = shape_file_str

    Config.record_log = config.getoption("--record") == "log"
    if Config.record_log:
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]

        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )


BUILTIN_MARKS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "timeout",
    "tryfirst",
    "trylast",
}


@pytest.fixture(scope="session", autouse=True)
def setup_once(request):
    if request.config.getoption("--query"):
        print("\nThis is query mode; all benchmark functions will be skipped.")
    # else:
    #     note_info = (
    #         "\n\nNote: The 'size' field below is for backward compatibility with previous versions of the benchmark. "
    #         "\nThis field will be removed in a future release."
    #     )
    #     print(note_info)


@pytest.fixture(scope="function", autouse=True)
def clear_function_cache():
    yield
    torch_device_fn.empty_cache()


@pytest.fixture(scope="module", autouse=True)
def clear_module_cache():
    yield
    torch_device_fn.empty_cache()


@pytest.fixture()
def extract_and_log_op_attributes(request):
    print("")
    op_attributes = []

    # Extract the 'recommended_shapes' attribute from the pytest marker decoration.
    for mark in request.node.iter_markers():
        if mark.name in BUILTIN_MARKS:
            continue
        op_specified_shapes = mark.kwargs.get("recommended_shapes")
        shape_desc = mark.kwargs.get("shape_desc", "M, N")
        rec_core_shapes = get_recommended_shapes(mark.name, op_specified_shapes)

        if rec_core_shapes:
            attri = OperationAttribute(
                op_name=mark.name,
                recommended_core_shapes=rec_core_shapes,
                shape_desc=shape_desc,
            )
            print(attri)
            op_attributes.append(attri.to_dict())

    if request.config.getoption("--query"):
        # Skip the real benchmark functions
        pytest.skip("Skipping benchmark due to the query parameter.")

    yield

    if Config.record_log and op_attributes:
        logging.info(json.dumps(op_attributes, indent=2))
