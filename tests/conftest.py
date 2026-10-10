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

# TODO(Qiming): Try remove this line
# import torch  # noqa: F401
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
FIRST_PARAMETER_ONLY = False

device = flag_gems.device

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILE = "accuracy_result.json"


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

    parser.addoption(
        "--first-parameter-only",
        action="store_true",
        help="Run only the first parameter combination for each test function",
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
    global FIRST_PARAMETER_ONLY

    TEST_RESULTS.clear()

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }

    RECORD_LOG = config.getoption("--record") == "log"
    RECORD_JSON = config.getoption("--record") == "json"
    TO_CPU = config.getoption("--ref") == "cpu"
    QUICK_MODE = config.getoption("--quick") is True
    FIRST_PARAMETER_ONLY = config.getoption("--first-parameter-only") is True

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

    # Print info when first-parameter-only is enabled
    if FIRST_PARAMETER_ONLY:
        print(f"\n{'='*70}")
        print("🔬 FIRST-PARAMETER-ONLY MODE ENABLED")
        print(
            "   Each test function will run with only its first parameter combination"
        )

        print(f"{'='*70}\n")


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
        all_skipped = bool(TEST_RESULTS) and all(
            result.get("result") == "skipped" for result in TEST_RESULTS.values()
        )
        config._override_registry.restore_all(allow_unused=all_skipped)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    registry = item.config._override_registry
    before = registry.call_counts()
    yield
    after = registry.call_counts()
    TEST_RESULTS[item.nodeid]["candidate_calls"] = {
        name: count - before.get(name, 0) for name, count in after.items()
    }


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


def _item_has_bfloat16_dtype(item):
    """Check if a parameterized test item includes bfloat16 in its dtype parameter."""
    if not hasattr(item, "callspec") or item.callspec is None:
        return False

    for param_name, param_value in item.callspec.params.items():
        if "dtype" not in param_name.lower():
            continue
        # Match torch.bfloat16, string "bfloat16", numpy dtype, etc.
        if "bfloat16" in str(param_value).lower():
            return True
    return False


def pytest_collection_modifyitems(session, config, items):
    """
    Modify the collected test items before they are executed.

    This hook handles:
    1. --collect-marks: Collect all test marks and write to a YAML file
    2. --first-parameter-only: Keep only one parameter combination per test function,
       preferring the first one that uses bfloat16 dtype.
    """

    first_parameter_only = config.getoption("--first-parameter-only")
    collect_marks_file = config.getoption("--collect-marks")

    if first_parameter_only:
        # Group items by test function (module + class + function name)
        function_groups = {}

        for item in items:
            # For parameterized tests, remove the parameter part from nodeid
            # Example: "tests/test_add.py::test_add[dtype0-0.001-shape0]" -> "tests/test_add.py::test_add"
            base_key = item.nodeid
            if "[" in base_key:
                base_key = base_key.split("[")[0]

            if base_key not in function_groups:
                function_groups[base_key] = []
            function_groups[base_key].append(item)

        # Keep only one item per group, preferring bfloat16
        kept_items = []
        bfloat16_count = 0
        fallback_count = 0

        for base_key, group_items in function_groups.items():
            # Sort by nodeid to ensure deterministic selection
            group_items.sort(key=lambda x: x.nodeid)

            # Priority 1: first item that uses bfloat16 dtype
            selected = None
            for item in group_items:
                if _item_has_bfloat16_dtype(item):
                    selected = item
                    bfloat16_count += 1
                    break

            # Priority 2: fallback to the very first item in the group
            if selected is None:
                selected = group_items[0]
                fallback_count += 1

            kept_items.append(selected)

        original_count = len(items)
        items[:] = kept_items

        print(f"\n{'='*70}")
        print("🔬 FIRST-PARAMETER-ONLY MODE ENABLED (bfloat16 preferred)")
        print(f"   Original tests: {original_count}")
        print(f"   Running tests:  {len(kept_items)} (one per test function)")
        print(f"      ├─ bfloat16 preferred: {bfloat16_count}")
        print(f"      └─ fallback (no bfloat16): {fallback_count}")
        print(f"{'='*70}\n")

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
        return
