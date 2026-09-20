# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fcntl
import json
import logging
import os
from datetime import datetime

import pytest

import torch
import yaml

import flag_gems
from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

BUILTIN_MARKS = {
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "timeout",
    "tryfirst",
    "trylast",
    "usefixtures",
    "xfail",
}
REGISTERED_MARKS = []
TEST_RESULTS = {}
RUNTEST_INFO = {}
RECORD_LOG = False
RECORD_JSON = False
TO_CPU = False
QUICK_MODE = False

device = flag_gems.device

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILE = "accuracy_result.json"

# ---------------------------------------------------------------------------
# Reference-environment / backend test skips
#
# Some entries in the accuracy report are not FlagGems bugs but limitations of
# the reference side or of the vendor runtime: the CPU reference torch build may
# lack LAPACK, lack cuDNN, or lack a CPU kernel for a narrow/complex dtype, and
# some aten ops are vendor-specific. Classifying them here keeps the report
# focused on real FlagGems accuracy issues.
# ---------------------------------------------------------------------------

# Ops the vendor runtime cannot run at all (skipped unconditionally on that
# vendor, regardless of --ref). Empty for now: cudnn_convolution_transpose used
# to live here, but FlagGems has its own Triton transpose-conv kernel and its
# test compares against the generic conv_transpose2d instead of the cuDNN-only
# aten entry point, so it runs on ROCm/MIOpen without cuDNN.
VENDOR_UNSUPPORTED_OPS = {}

# (op marker -> dtypes) with no CPU reference kernel. These pass against a GPU
# reference, so only skip them when the reference runs on CPU (--ref cpu).
CPU_REF_UNSUPPORTED_DTYPES = {
    "flipud": {torch.complex32},  # aten flip_cpu has no ComplexHalf kernel
    "binomial": {torch.float16},  # aten binomial_cpu has no Half kernel
}

# Substrings of a reference-side RuntimeError meaning the CPU reference build
# cannot compute the ground truth (missing LAPACK / cuDNN). Under --ref cpu such
# a failure is a reference-environment limitation, not a FlagGems bug. Matching
# on the message keeps this per-parameter precise (e.g. linalg.norm only needs
# LAPACK for ord in {2, -2, nuc}), so it never hides a real FlagGems failure.
CPU_REF_ENV_ERROR_SUBSTRINGS = (
    "requires compiling PyTorch with LAPACK",
    "LAPACK library not found",
    "not compiled with cuDNN",
)


def _apply_backend_skips(config, items):
    vendor = getattr(flag_gems, "vendor_name", None)
    cpu_ref = config.getoption("--ref") == "cpu"
    vendor_unsupported = VENDOR_UNSUPPORTED_OPS.get(vendor, set())
    for item in items:
        mark_names = {mark.name for mark in item.iter_markers()}

        blocked = mark_names & vendor_unsupported
        if blocked:
            op = sorted(blocked)[0]
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"{op}: unsupported on '{vendor}' backend "
                        "(vendor runtime lacks the required library, e.g. cuDNN)"
                    )
                )
            )
            continue

        if not cpu_ref:
            continue
        params = getattr(getattr(item, "callspec", None), "params", {}) or {}
        dtype = params.get("dtype")
        if dtype is None:
            continue
        for op in mark_names:
            if dtype in CPU_REF_UNSUPPORTED_DTYPES.get(op, set()):
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{op}: CPU reference has no kernel for {dtype} "
                            "(--ref cpu); passes with a GPU reference"
                        )
                    )
                )
                break


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default=device,
        required=False,
        choices=[device, "cpu"],
        help="device to run reference tests on",
    )

    parser.addoption(
        "--quick",
        action="store_true",
        help="run tests on quick mode",
    )

    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            required=False,
            choices=["none", "log", "json"],
            help="record test results in log/json files or not",
        )
        parser.addoption(
            "--output",
            help="path to the result file",
        )

    except ValueError:
        # Mixed test+benchmark pytest runs may already register --record in
        # benchmark/conftest.py. Reuse the existing option in that case.
        pass

    try:
        parser.addoption(
            "--collect-marks",
            default=None,
            help="Collect the tests with marker information and write to the specified file",
        )
    except ValueError:
        pass

    # Add dynamic operator override options
    add_override_arguments(parser)


def pytest_configure(config):
    global RECORD_LOG
    global RECORD_JSON
    global REPORT_FILE
    global REGISTERED_MARKS
    global RUNTEST_INFO
    global TO_CPU
    global QUICK_MODE

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }

    RECORD_LOG = config.getoption("--record") == "log"
    RECORD_JSON = config.getoption("--record") == "json"
    TO_CPU = config.getoption("--ref") == "cpu"
    QUICK_MODE = config.getoption("--quick") is True

    if RECORD_JSON:
        report_file = config.getoption("--output")
        if report_file:
            REPORT_FILE = report_file

    if RECORD_LOG:
        RUNTEST_INFO = {}
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

    # Apply dynamic operator overrides
    config._override_registry = apply_overrides_from_args(config.option)


def pytest_runtest_teardown(item, nextitem):
    if not RECORD_LOG:
        return

    if hasattr(item, "callspec"):
        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKS
        ]
        if len(op_marks) > 0:
            params = str(item.callspec.params)
            for op_mark in op_marks:
                if op_mark not in RUNTEST_INFO:
                    RUNTEST_INFO[op_mark] = [params]
                else:
                    RUNTEST_INFO[op_mark].append(params)
        else:
            func_name = item.function.__name__
            logging.warning("There is no mark at {}".format(func_name))


def pytest_sessionfinish(session, exitstatus):
    if RECORD_LOG:
        logging.info(json.dumps(RUNTEST_INFO, indent=2))


def pytest_unconfigure(config):
    """Cleanup: restore all overridden operators."""
    if hasattr(config, "_override_registry"):
        config._override_registry.restore_all()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    TEST_RESULTS[item.nodeid] = {"params": None, "result": None, "opname": None}
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params

    TEST_RESULTS[item.nodeid]["params"] = param_values
    # get all mark
    all_marks = [mark.name for mark in item.iter_markers()]
    # exclude marks，such as parametrize、skipif and so on
    operator_marks = [mark for mark in all_marks if mark not in BUILTIN_MARKS]
    TEST_RESULTS[item.nodeid]["opname"] = operator_marks


def get_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    elif isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    else:
        return str(report.longrepr)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    result = TEST_RESULTS.setdefault(
        report.nodeid, {"params": None, "result": None, "opname": None}
    )
    if report.when == "setup":
        if report.outcome == "skipped":
            reason = get_reason(report)
            result["result"] = "skipped"
            result["reason"] = reason
    elif report.when == "call":
        result["result"] = report.outcome
        if report.outcome in ["skipped", "failed"]:
            reason = get_reason(report)
            result["reason"] = reason
        else:
            result["reason"] = None


def pytest_terminal_summary(terminalreporter):
    data = TEST_RESULTS
    with open(REPORT_FILE, "a+") as json_file:
        fcntl.flock(json_file, fcntl.LOCK_EX)
        json_file.seek(0)
        content = json_file.read()
        if content:
            existing_data = json.loads(content)
            existing_data.update(TEST_RESULTS)
            data = existing_data
        json_file.seek(0)
        json_file.truncate()
        json.dump(data, json_file, indent=2, default=str)
        json_file.flush()
        os.fsync(json_file.fileno())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # Under --ref cpu, a reference-side RuntimeError from a torch build missing
    # LAPACK/cuDNN is an environment limitation, not a FlagGems accuracy bug.
    # Report it as skipped instead of failed so the accuracy report stays clean.
    if report.when == "call" and report.failed and TO_CPU and call.excinfo is not None:
        message = str(call.excinfo.value)
        if any(s in message for s in CPU_REF_ENV_ERROR_SUBSTRINGS):
            report.outcome = "skipped"
            first_line = message.splitlines()[0] if message else ""
            report.longrepr = (
                str(item.fspath),
                item.location[1] or 0,
                "Skipped: CPU reference environment cannot compute ground truth: "
                + first_line,
            )


def pytest_collection_modifyitems(session, config, items):
    _apply_backend_skips(config, items)

    collect_marks_file = config.getoption("--collect-marks")
    if collect_marks_file:
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

        with open(collect_marks_file, "w") as f:
            yaml.dump(report, f, indent=2)

        # Skip all tests
        items.clear()
