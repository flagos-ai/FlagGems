import json
import logging
import os

import pytest
import torch
import yaml

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import consts

device = flag_gems.device
vendor_name = flag_gems.vendor_name
recordLogger = logging.getLogger("flag_gems_benchmark")
recordLogger.propagate = False
Config = None

BUILTIN_MARKS = (
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "timeout",
    "tryfirst",
    "trylast",
)
REGISTERED_MARKS = []
TEST_RESULTS = {}
REPORT_FILE = "benchmark_result.json"


def update_result(op, data):
    if not Config.record_json:
        return

    TEST_RESULTS.setdefault(op, {})
    TEST_RESULTS[op].setdefault("details", [])
    TEST_RESULTS[op]["details"].append(data)


def emit_record_logger(message: str) -> None:
    if not Config.record_log:
        return

    if recordLogger.handlers:
        handler = recordLogger.handlers[0]
        if getattr(handler, "stream", None) is None:
            handler.acquire()
            try:
                handler.stream = handler._open()
            finally:
                handler.release()

    recordLogger.info(message)


class BenchConfig:
    def __init__(self):
        self.mode = consts.BenchMode.KERNEL
        self.bench_level = consts.BenchLevel.COMPREHENSIVE
        self.warm_up = consts.DEFAULT_WARMUP_COUNT
        self.repetition = consts.DEFAULT_ITER_COUNT

        # Speed Up Benchmark Test, Big Shape Will Cause Timeout
        if vendor_name == "kunlunxin":
            self.warm_up = 1
            self.repetition = 1

        self.record_log = False
        self.record_json = False
        self.user_desired_dtypes = None
        self.user_desired_metrics = None
        self.shape_file = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
        self.query = False
        self.parallel = 0


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
            "Specify how to measure latency, 'kernel' for device kernel, "
            "'operator' for end2end operator or 'wrapper' for runtime wrapper."
        ),
    )

    parser.addoption(
        "--level",
        action="store",
        default="comprehensive",
        required=False,
        choices=[level.value for level in consts.BenchLevel],
        help="Specify the benchmark level: comprehensive, or core.",
    )

    parser.addoption(
        "--warmup",
        default=consts.DEFAULT_WARMUP_COUNT,
        help="Number of warmup runs before benchmark run.",
    )

    parser.addoption(
        "--iter",
        default=consts.DEFAULT_ITER_COUNT,
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
        choices=consts.ALL_AVAILABLE_METRICS,
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
            for ele in consts.FLOAT_DTYPES
            + consts.INT_DTYPES
            + consts.BOOL_DTYPES
            + [torch.cfloat]
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

    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            required=False,
            choices=["none", "log", "json"],
            help="Benchmark info recorded in log/json files or not",
        )
        parser.addoption(
            "--output",
            default=REPORT_FILE,
            help="Path to report file for JSON output",
        )
    except ValueError:
        # Mixed test+benchmark pytest runs may already register --record in
        # tests/conftest.py. Reuse the existing option in that case.
        pass

    parser.addoption(
        "--parallel",
        action="store",
        type=int,
        default=0,
        help=(
            "Enable multi-GPU parallel benchmark execution across shapes. "
            "Example: --parallel 8 means using GPU 0~7 in parallel. "
            "Default 0 means serial execution."
        ),
    )

    try:
        parser.addoption(
            "--collect-marks",
            action="store_true",
            help="Collect the tests with marker information without executing them",
        )
    except ValueError:
        # Mixed test+benchmark pytest runs may already register this option in
        # tests/conftest.py. Reuse the existing option in that case.
        pass


def pytest_configure(config):
    global Config  # noqa: F824
    global REPORT_FILE
    global REGISTERED_MARKS

    Config = BenchConfig()

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }

    mode_value = config.getoption(
        "--mode" if vendor_name != "kunlunxin" else "--fg_mode"
    )
    Config.mode = consts.BenchMode(mode_value)

    Config.query = config.getoption("--query")

    level_value = config.getoption("--level")
    Config.bench_level = consts.BenchLevel(level_value)

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
    Config.record_json = config.getoption("--record") == "json"

    Config.parallel = int(config.getoption("--parallel") or 0)
    if Config.record_json:
        Config.output = config.getoption("--output")
        REPORT_FILE = Config.output

    if Config.record_log:
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]

        log_file = "result_{}.log".format("_".join(cmd_args)).replace("_-", "-")

        for h in list(recordLogger.handlers):
            recordLogger.removeHandler(h)
            try:
                h.close()
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to close handler: {e}")

        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8", delay=False)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        recordLogger.addHandler(handler)
        recordLogger.setLevel(logging.INFO)
        emit_record_logger("Benchmark record logger enabled")


@pytest.fixture(scope="session", autouse=True)
def setup_once(request):
    if request.config.getoption("--query"):
        print("\nThis is query mode; all benchmark functions will be skipped.")


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
        rec_core_shapes = consts.get_recommended_shapes(mark.name, op_specified_shapes)

        if rec_core_shapes:
            attri = consts.OperationAttribute(
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
        emit_record_logger(json.dumps(op_attributes, indent=2))


def get_reason(report):
    """Get reason for skipped or failed test."""

    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message

    if isinstance(report.longrepr, tuple):
        return report.longrepr[2]

    return str(report.longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    out = yield
    report = out.get_result()
    all_marks = [mark.name for mark in item.iter_markers()]
    # exclude builtin marks
    marks = [mark for mark in all_marks if mark not in BUILTIN_MARKS]
    # Assume the first mark is the operator's ID
    opid = marks[0] if marks else item.nodeid
    # Set the operator ID for the next function to use
    report.opid = opid


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if not Config.record_json:
        return

    op = report.opid
    TEST_RESULTS.setdefault(op, {})

    if report.when == "setup":
        if report.outcome == "skipped":
            reason = get_reason(report)
            TEST_RESULTS[op]["result"] = "skipped"
            TEST_RESULTS[op]["reason"] = reason
            TEST_RESULTS[op]["test_case"] = report.nodeid

    elif report.when == "call":
        TEST_RESULTS[op]["result"] = report.outcome
        TEST_RESULTS[op]["test_case"] = report.nodeid

        if report.outcome in ["skipped", "failed"]:
            reason = get_reason(report)
            TEST_RESULTS[op]["reason"] = reason
        else:
            TEST_RESULTS[op]["reason"] = None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Combine and dump the result into JSON."""
    if not Config.record_json:
        return

    data = TEST_RESULTS
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as f:
            existing_data = json.load(f)
        existing_data.update(TEST_RESULTS)
        data = existing_data

    with open(REPORT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def pytest_collection_modifyitems(session, config, items):
    if not config.getoption("--collect-marks"):
        return

    report = []
    for item in items:
        data = {}

        # Collect some general information
        if item.cls:
            data["class"] = item.cls.__name__
        data["test_case"] = item.name
        if item.originalname:
            data["function"] = item.originalname
        data["file"] = item.location[0]

        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKS
        ]

        data["marks"] = op_marks
        report.append(data)

    print(yaml.dump(report, indent=2))

    # Skip all tests
    items.clear()
