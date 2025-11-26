import json
import logging
import os
from datetime import datetime

import pytest

import flag_gems
import torch

import itertools
from .skipped_nodeid import UNSELECT_NODEIDS
from _pytest.python import Metafunc
device = flag_gems.device

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"test_detail_and_result_{timestamp}.json"


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
        (
            "--mode" if flag_gems.vendor_name != "kunlunxin" else "--fg_mode"
        ),  # TODO: fix pytest-* common --mode args,
        action="store",
        default="normal",
        required=False,
        choices=["normal", "quick"],
        help="run tests on normal or quick mode",
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
        "--limit-cases",
        type=int, 
        default=None,
    )
    parser.addoption(
        "--compiler-choice",
        action="store",
        default="none",
        required=False,
        choices=["none", "flagtree"],
        help="compiler to run  tests with",
    )
    parser.addoption(
    "--skipped-unselected-nodeids",
    action="store_true",
    default=False,
    help="Whether to use the unselected nodeids list.", 
    )


def pytest_configure(config):
    global TO_CPU
    TO_CPU = config.getoption("--ref") == "cpu"

    global QUICK_MODE
    QUICK_MODE = config.getoption("--mode") == "quick"

    global LIMIT
    LIMIT = config.getoption("--limit-cases")

    global COMPILER_CHOICE
    COMPILER_CHOICE = config.getoption("--compiler-choice") == "flagtree"

    global SKIP_SPEC_NODEIDS
    SKIP_SPEC_NODEIDS = config.getoption("--skipped-unselected-nodeids")


    global RECORD_LOG
    RECORD_LOG = config.getoption("--record") == "log"
    if RECORD_LOG:
        global RUNTEST_INFO, BUILTIN_MARKS, REGISTERED_MARKERS
        RUNTEST_INFO = {}
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
        REGISTERED_MARKERS = {
            marker.split(":")[0].strip() for marker in config.getini("markers")
        }
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
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKERS
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


test_results = {}

CUSTOM_TEST_PARAMS = {
    "test_accuracy_dropout": {
        "shape": [(2, 19, 7)],
        "p": [0.1],
        "dtype": [torch.float16],
    },
}

CUSTOM_PARAM_MARK = pytest.mark.custom_params

@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: Metafunc) -> None:
    test_name = metafunc.function.__name__

    if test_name not in CUSTOM_TEST_PARAMS:
        return

    cfg = CUSTOM_TEST_PARAMS[test_name]

    argnames = list(cfg.keys())  
    value_lists = [cfg[name] for name in argnames]
    combos = list(itertools.product(*value_lists))

    params = [
        pytest.param(*vals, marks=CUSTOM_PARAM_MARK)
        for vals in combos
    ]


    metafunc._calls = []

    metafunc.parametrize(argnames, params)
    nodes = [
        metafunc.definition,
        getattr(metafunc, "cls", None),
        getattr(metafunc, "module", None),
    ]
    for node in nodes:
        if node is None:
            continue
        if hasattr(node, "own_markers"):
            node.own_markers = [
                m for m in node.own_markers if m.name != "parametrize"
            ]

    return

@pytest.hookimpl
def pytest_collection_modifyitems(config, items):
    selected = []
    deselected = []
    counter = {}
    if not COMPILER_CHOICE and not LIMIT:
        return 

    for item in items:
        if COMPILER_CHOICE:
            if item.get_closest_marker("custom_params"):
                selected.append(item)
            else:
                deselected.append(item)
        if SKIP_SPEC_NODEIDS:
            if item.nodeid in UNSELECT_NODEIDS:
                deselected.append(item)
        if LIMIT:
            func_name = item.function.__name__
            counter.setdefault(func_name, 0)

            if counter[func_name] < LIMIT:
                selected.append(item)
                counter[func_name] += 1
            else:
                deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
        
    test_results[item.nodeid] = {"params": None, "result": None, "opname": None}
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params

    test_results[item.nodeid]["params"] = param_values

    all_marks = [mark.name for mark in item.iter_markers()]

    exclude_marks = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "inplace"}
    operator_marks = [mark for mark in all_marks if mark not in exclude_marks]
    test_results[item.nodeid]["opname"] = operator_marks


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == "call":
        test_results[report.nodeid]["result"] = report.outcome


def pytest_terminal_summary(terminalreporter):
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(test_results)
    else:
        existing_data = test_results

    with open("result.json", "w") as json_file:
        json.dump(existing_data, json_file, indent=4, default=str)
