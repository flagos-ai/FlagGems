import json
import logging
import os
from datetime import datetime

import pytest

# TODO(Qiming): Try remove this line
import torch  # noqa: F401
import yaml

import flag_gems

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
REGISTERED_MARKS = []
TEST_RESULTS = {}
RUNTEST_INFO = {}
RECORD_LOG = False
TO_CPU = False
QUICK_MODE = False

device = flag_gems.device

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILE = f"report_{TIMESTAMP}.json"


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
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="tests function param recorded in log files or not",
    )

    parser.addoption(
        "--collect-marks",
        action="store_true",
        help="Collect the tests with marker information without executing them",
    )


def pytest_configure(config):
    global RECORD_LOG
    global REGISTERED_MARKS
    global RUNTEST_INFO
    global TO_CPU
    global QUICK_MODE

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }

    RECORD_LOG = config.getoption("--record") == "log"
    TO_CPU = config.getoption("--ref") == "cpu"
    QUICK_MODE = config.getoption("--quick") is True

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
    if report.when == "setup":
        if report.outcome == "skipped":
            reason = get_reason(report)
            TEST_RESULTS[report.nodeid]["result"] = "skipped"
            TEST_RESULTS[report.nodeid]["reason"] = reason
    elif report.when == "call":
        TEST_RESULTS[report.nodeid]["result"] = report.outcome
        if report.outcome in ["skipped", "failed"]:
            reason = get_reason(report)
            TEST_RESULTS[report.nodeid]["reason"] = reason
        else:
            TEST_RESULTS[report.nodeid]["reason"] = None


def pytest_terminal_summary(terminalreporter):
    data = TEST_RESULTS
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(TEST_RESULTS)
        data = existing_data

    with open(f"result_{TIMESTAMP}.json", "w") as json_file:
        json.dump(data, json_file, indent=2, default=str)


def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--collect-marks"):
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
