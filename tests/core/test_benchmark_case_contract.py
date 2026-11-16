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

"""Host-only checks: no FlagGems, Torch, Triton or device imports required.

Run with PYTHONPATH=. pytest --confcutdir=tests/core tests/core/test_benchmark_case_contract.py
"""

import json

import pytest

from benchmark.cases import BenchmarkCaseList, BenchmarkCasePlan, BenchmarkCaseSpec


def test_case_list_contract_excludes_private_materialization_state():
    plan = BenchmarkCasePlan(shape={"input": (4, 8)}, params={"dim": -1})
    case = BenchmarkCaseSpec(
        case_id="benchmark/test_example.py::test_example::core::float32::0",
        ordinal=0,
        dtype="torch.float32",
        shape=plan.shape,
        params=plan.params,
        builder_args=(plan, object()),
    )
    report = BenchmarkCaseList(op_name="example", level="core", cases=(case,))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["schema_version"] == "flaggems.benchmark-case-list/v2"
    assert payload["phase"] == "timing"
    assert payload["cases"] == [
        {
            "case_id": case.case_id,
            "ordinal": 0,
            "dtype": "torch.float32",
            "shape": {"input": [4, 8]},
            "params": {"dim": -1},
        }
    ]


@pytest.mark.parametrize(
    "shape", [{"input": ()}, {"input": (0, 8)}, {"a": (4,), "b": (1,)}]
)
def test_shapes_and_parameters_roundtrip(shape):
    case = BenchmarkCaseSpec(
        "node::core::float16::0",
        0,
        "torch.float16",
        shape,
        {"dim": -1, "keepdim": False, "out": None},
    )
    encoded = json.loads(json.dumps(case.to_dict()))
    assert encoded["params"] == {"dim": -1, "keepdim": False, "out": None}
    assert encoded["shape"] == json.loads(json.dumps(shape))


def test_metadata_does_not_silently_stringify_non_json_objects():
    case = BenchmarkCaseSpec("case", 0, "torch.float32", {"input": object()})
    with pytest.raises(TypeError):
        json.dumps(case.to_dict())


def test_plan_defaults_are_not_shared():
    first = BenchmarkCasePlan({"input": (4,)})
    second = BenchmarkCasePlan({"input": (8,)})
    first.params["dim"] = 0
    assert second.params == {}
