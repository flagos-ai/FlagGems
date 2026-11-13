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

"""Synthetic format/contract checks; these are not collected NVIDIA timings."""

import copy
import json
from types import SimpleNamespace

import pytest

from benchmark.baseline import BaselineData
from tools.performance_report import parse_perf_data

DTYPE = "torch.float32"
TENSOR = {"shape": [4, 8], "dtype": DTYPE, "stride": [8, 1], "requires_grad": False}
UNARY_CALL = {"args": [TENSOR], "kwargs": {}}
CALL = {"args": [TENSOR, 1], "kwargs": {"keepdim": False}}


def baseline(record=None, key="case0", op="reduce"):
    if record is None:
        record = {"latency_ms": 4.0, "config": {"inputs": copy.deepcopy(CALL)}}
    return BaselineData({op: {"shapes": {key: {DTYPE: record}}}}, "synthetic.json")


def test_nested_metadata_matches_even_when_shape_key_is_a_descriptor():
    call = {"args": [[TENSOR, TENSOR]], "kwargs": {"dim": 0}}
    data = baseline({"latency_ms": 4, "config": {"inputs": copy.deepcopy(call)}})
    assert data.lookup("reduce", DTYPE, call)[1:] == ("case0", None)
    call["kwargs"]["dim"] = 1
    assert data.lookup("reduce", DTYPE, call)[0] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [8, 4]),
        ("dtype", "torch.int64"),
        ("stride", [1, 4]),
        ("requires_grad", True),
    ],
)
def test_tensor_identity_mismatch_does_not_use_first_shape_shortcut(field, value):
    call = copy.deepcopy(CALL)
    call["args"][0][field] = value
    assert baseline(key="[4, 8]").lookup("reduce", DTYPE, call)[0] is None


@pytest.mark.parametrize("value", [0, True, 2, 1.0])
def test_scalar_type_and_value_mismatches_fall_back(value):
    call = copy.deepcopy(CALL)
    call["args"][1] = value
    assert baseline().lookup("reduce", DTYPE, call)[0] is None


def test_ambiguity_does_not_select_first_record():
    data = baseline()
    shapes = data.data["reduce"]["shapes"]
    shapes["case1"] = copy.deepcopy(shapes["case0"])
    assert data.lookup("reduce", DTYPE, CALL)[2] == "ambiguous_match"


def test_named_inputs_require_binding_and_compare_scalars():
    record = {
        "latency_ms": 1,
        "config": {
            "inputs": {
                "x": TENSOR,
                "dim": {"scalar": 1},
            }
        },
    }
    data = baseline(record)
    assert data.lookup("reduce", DTYPE, CALL)[0] is None
    assert data.lookup("reduce", DTYPE, CALL, {"x": TENSOR, "dim": 1})[0]
    assert data.lookup("reduce", DTYPE, CALL, {"x": TENSOR, "dim": 0})[0] is None


def test_legacy_shape_and_microsecond_units_and_private_operator_names():
    data = baseline({"T_us": 500}, key="[4,8]", op="_reduce")
    assert data.lookup("_reduce", DTYPE, UNARY_CALL)[0]["latency_ms"] == 0.5
    assert data.lookup("_reduce", "torch.float16", CALL)[0] is None
    assert data.lookup("missing", DTYPE, CALL)[2] == "operator_missing"


@pytest.mark.parametrize(
    "latency", [None, 0, -1, True, "4", float("nan"), float("inf")]
)
def test_invalid_latency_is_a_miss(latency):
    data = baseline({"latency_ms": latency}, key="[4, 8]")
    assert data.lookup("reduce", DTYPE, UNARY_CALL)[2] == "invalid_latency"


@pytest.mark.parametrize("payload", ["[]", "null", "{broken"])
def test_invalid_file_warns_and_disables_comparison(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    with pytest.warns(UserWarning, match="using native baseline"):
        assert BaselineData.load(path) is None
    assert BaselineData.load(None) is None


def test_missing_file_warns_and_valid_metadata_is_preserved(tmp_path):
    with pytest.warns(UserWarning):
        assert BaselineData.load(tmp_path / "missing") is None
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"_environment": {}, "_reduce": {"shapes": {}}}))
    assert "_reduce" in BaselineData.load(path).data


def with_scaling(data):
    data.data["_scaling_factors"] = {
        "chips": [
            {
                "vendor": "nvidia",
                "chip": "H20",
                "measured": {"bandwidth_gbps": 0.5, "vector": {"fp32": 0.25}},
                "nominal": {"tensor": {"fp32": 0.125}},
            },
            {"vendor": "nvidia", "chip": "H800", "measured": {"bandwidth_gbps": 1}},
        ]
    }
    return data


@pytest.mark.parametrize(
    "unit,expected", [("mem", 0.5), ("cuda", 0.25), ("tensor", 0.125)]
)
def test_scaling_selects_chip_and_resource_and_prefers_measured(unit, expected):
    data = with_scaling(baseline())
    assert data.scaling_factor("nvidia", "NVIDIA H20-3e", unit, DTYPE) == expected
    assert data.scaling_factor("nvidia", "NVIDIA H800", "mem", DTYPE) == 1
    assert data.scaling_factor("nvidia", "NVIDIA H100", unit, DTYPE) is None
    assert data.scaling_factor("nvidia", "", unit, DTYPE) is None
    assert data.scaling_factor("ascend", "NVIDIA H20", unit, DTYPE) is None


@pytest.mark.parametrize("factor", [-1, 0, True, float("inf"), "0.5"])
def test_invalid_scaling_falls_back_to_raw_ratio(factor):
    data = with_scaling(baseline())
    data.data["_scaling_factors"]["chips"][0]["measured"]["bandwidth_gbps"] = factor
    assert data.scaling_factor("nvidia", "NVIDIA H20", "mem", DTYPE) is None


def test_external_ratio_preserves_native_measurements_and_records_provenance():
    data = with_scaling(
        baseline(
            {"latency_ms": 4, "bottle_neck_unit": "mem", "config": {"inputs": CALL}}
        )
    )
    metric = SimpleNamespace(latency=2.0, latency_base=6.0, speedup=3.0)
    data.apply(metric, "reduce", DTYPE, CALL, None, "nvidia", "NVIDIA H20")
    assert metric.compared_speedup == 4.0
    assert metric.base_raw_speedup == 2.0
    assert metric.base_scaling_factor == 0.5
    assert metric.base_data_source == "synthetic.json"
    assert metric.latency_base == 6.0 and metric.speedup == 3.0
    assert metric.base_miss_reason is None


def test_skip_native_can_use_external_data_and_a_miss_keeps_na():
    metric = SimpleNamespace(latency=2.0, latency_base=None, speedup=None)
    data = baseline()
    data.apply(metric, "reduce", DTYPE, CALL, None, "unknown", "")
    assert metric.compared_speedup == 2.0 and metric.speedup is None
    missing = SimpleNamespace(latency=2.0, speedup=None)
    data.apply(missing, "missing", DTYPE, CALL, None, "unknown", "")
    assert missing.speedup is None and not hasattr(missing, "compared_speedup")


@pytest.mark.parametrize("skip_native", [False, True])
def test_real_summary_prefers_external_per_case_and_preserves_fallback(
    tmp_path, skip_native
):
    item = {
        "dtype": DTYPE,
        "result": [
            {
                "shape_detail": [4],
                "latency_base": 6,
                "latency": 2,
                "speedup": 3,
                "compared_speedup": 4,
                "base_latency_ms": 8,
            },
            {
                "shape_detail": [8],
                "latency_base": 2,
                "latency": 1,
                "speedup": 2,
                "base_miss_reason": "shape_or_dtype_missing",
            },
        ],
    }
    if skip_native:
        item["native_baseline_skip_reason"] = "native unavailable"
        item["result"][1]["speedup"] = None
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"reduce": {"result": "passed", "details": [item]}}))
    result = parse_perf_data("reduce", path)["data"]["fp32"]
    assert result["speedup"] == (4 if skip_native else 3)
    assert result["details"]["[4]"]["native_speedup"] == 3
    assert result["details"]["[8]"]["base_miss_reason"] == "shape_or_dtype_missing"


@pytest.mark.parametrize("config", [[], "invalid", {"inputs": []}, {"inputs": None}])
def test_malformed_metadata_does_not_fall_through_to_legacy_shape(config):
    data = baseline({"latency_ms": 4, "config": config}, key="[4, 8]")
    assert data.lookup("reduce", DTYPE, CALL)[0] is None


def test_malformed_resource_and_oversized_latency_do_not_break_benchmarks():
    data = with_scaling(baseline())
    assert data.scaling_factor("nvidia", "NVIDIA H20", [], DTYPE) is None
    data = baseline({"latency_ms": 10**1000}, key="[4, 8]")
    assert data.lookup("reduce", DTYPE, UNARY_CALL)[2] == "invalid_latency"


@pytest.mark.parametrize("target", ["npu", "mlu"])
def test_factory_device_matches_selected_backend_without_weakening_other_inputs(target):
    source = {
        "args": [],
        "kwargs": {"size": [4, 8], "fill_value": 3.0, "dtype": DTYPE, "device": "cuda"},
    }
    call = copy.deepcopy(source)
    call["kwargs"]["device"] = target
    data = baseline({"latency_ms": 4, "config": {"inputs": source}}, op="full")
    assert data.lookup("full", DTYPE, call, target_device=target)[
        0
    ], "factory target device must not prevent a match"


def test_thead_runtime_vendor_uses_ops_baseline_t_head_scaling():
    data = baseline()
    data.data["_scaling_factors"] = {
        "chips": [
            {"vendor": "t-head", "chip": "m890", "measured": {"bandwidth_gbps": 0.78}}
        ]
    }
    assert data.scaling_factor("thead", "M890", "mem", DTYPE) == 0.78


def test_h200_does_not_use_h20_scaling():
    data = with_scaling(baseline())
    assert data.scaling_factor("nvidia", "NVIDIA H200", "mem", DTYPE) is None


@pytest.mark.parametrize(
    "source_device,actual_device,target,matched",
    [
        ("cuda:0", "npu:0", "npu", True),
        ("cuda", "mlu:2", "mlu:2", True),
        ("cuda", "npu:1", "npu", False),
        ("cuda", "cpu", "npu", False),
        ("cpu", "npu", "npu", False),
        ("cuda:1", "npu", "npu", False),
        ("cuda", "npu", None, False),
        ("cuda", "npu", "mlu", False),
    ],
)
def test_device_normalization_requires_the_selected_target(
    source_device, actual_device, target, matched
):
    source = {"args": [], "kwargs": {"device": source_device, "size": [4, 8]}}
    call = copy.deepcopy(source)
    call["kwargs"]["device"] = actual_device
    data = baseline({"latency_ms": 4, "config": {"inputs": source}})
    result = data.lookup("reduce", DTYPE, call, target_device=target)
    assert (result[0] is not None) == matched


@pytest.mark.parametrize(
    "changed_key,changed_value",
    [
        ("size", [8, 4]),
        ("fill_value", 4.0),
        ("dtype", "torch.float16"),
        ("auxiliary_device", "npu"),
    ],
)
def test_factory_normalization_still_checks_all_other_parameters(
    changed_key, changed_value
):
    source = {
        "args": [],
        "kwargs": {
            "size": [4, 8],
            "fill_value": 3.0,
            "dtype": DTYPE,
            "device": "cuda",
            "auxiliary_device": "cpu",
        },
    }
    call = copy.deepcopy(source)
    call["kwargs"]["device"] = "npu"
    call["kwargs"][changed_key] = changed_value
    data = baseline({"latency_ms": 4, "config": {"inputs": source}})
    assert data.lookup("reduce", DTYPE, call, target_device="npu")[0] is None
    assert data.data["reduce"]["shapes"]["case0"][DTYPE]["config"]["inputs"] == source


def test_named_device_metadata_uses_the_same_target_rule():
    inputs = {"size": {"scalar": [4, 8]}, "device": {"scalar": "cuda"}}
    data = baseline({"latency_ms": 4, "config": {"inputs": inputs}})
    named = {"size": [4, 8], "device": "mlu"}
    call = {"args": [], "kwargs": named}
    assert data.lookup("reduce", DTYPE, call, named, target_device="mlu")[0]
    assert data.lookup("reduce", DTYPE, call, named, target_device="npu")[0] is None


def test_overlapping_models_select_the_exact_chip_and_preserve_suffixes():
    data = with_scaling(baseline())
    data.data["_scaling_factors"]["chips"].append(
        {
            "vendor": "nvidia",
            "chip": "H200",
            "measured": {"bandwidth_gbps": 0.8},
        }
    )
    assert data.scaling_factor("nvidia", "NVIDIA H200", "mem", DTYPE) == 0.8
    assert data.scaling_factor("nvidia", "NVIDIA H20-3e", "mem", DTYPE) == 0.5
    assert data.scaling_factor("nvidia", "NVIDIA XH20", "mem", DTYPE) is None
    assert data.scaling_factor("nvidia", "NVIDIA H20X", "mem", DTYPE) is None


@pytest.mark.parametrize(
    "runtime_vendor,stored_vendor", [("thead", "t-head"), ("t-head", "thead")]
)
def test_vendor_alias_is_explicit_and_does_not_match_other_vendors(
    runtime_vendor, stored_vendor
):
    data = baseline()
    data.data["_scaling_factors"] = {
        "chips": [
            {
                "vendor": stored_vendor,
                "chip": "m890",
                "nominal": {"bandwidth_gbps": 0.78},
            },
        ]
    }
    assert data.scaling_factor(runtime_vendor, "M890", "mem", DTYPE) == 0.78
    assert data.scaling_factor("thead2", "M890", "mem", DTYPE) is None


@pytest.mark.parametrize("kwargs", [None, 1, "invalid", []])
def test_malformed_kwargs_metadata_falls_back_without_crashing(kwargs):
    inputs = {"args": CALL["args"], "kwargs": kwargs}
    data = baseline({"latency_ms": 4, "config": {"inputs": inputs}})
    assert data.lookup("reduce", DTYPE, CALL, target_device="npu")[0] is None


@pytest.mark.parametrize(
    "call",
    [
        {"args": [TENSOR, TENSOR], "kwargs": {}},
        {"args": [TENSOR, 1], "kwargs": {}},
        {"args": [TENSOR], "kwargs": {"dim": 1}},
        {"args": [[TENSOR]], "kwargs": {}},
    ],
)
def test_legacy_shape_cannot_establish_multi_input_or_parameter_abi(call):
    data = baseline({"latency_ms": 4}, key="[4, 8]")
    assert data.lookup("reduce", DTYPE, call)[0] is None


def test_legacy_sglang_packed_silu_and_mul_does_not_match_two_tensor_gems_abi():
    tensor = {"shape": [1, 2048], "dtype": "torch.bfloat16"}
    data = BaselineData(
        {
            "silu_and_mul": {
                "shapes": {"[1, 2048]": {"torch.bfloat16": {"latency_ms": 0.005}}}
            }
        },
        "legacy-sglang.json",
    )
    call = {"args": [tensor, tensor], "kwargs": {}}
    assert data.lookup("silu_and_mul", "torch.bfloat16", call)[0] is None


@pytest.mark.parametrize("inputs", [{}, {"x": TENSOR}])
@pytest.mark.parametrize("dim", [0, 1])
def test_incomplete_named_metadata_cannot_choose_a_semantic_baseline(inputs, dim):
    data = baseline({"latency_ms": 8, "config": {"inputs": inputs}})
    call = {"args": [TENSOR, dim], "kwargs": {}}
    assert data.lookup("reduce", DTYPE, call, {"x": TENSOR, "dim": dim})[0] is None


def test_complete_named_metadata_still_matches_and_checks_all_parameters():
    data = baseline(
        {
            "latency_ms": 8,
            "config": {
                "inputs": {"x": TENSOR, "dim": {"scalar": 1}, "out": {"scalar": None}}
            },
        }
    )
    named = {"x": TENSOR, "dim": 1, "out": None}
    assert data.lookup("reduce", DTYPE, CALL, named)[0] is not None
    named["dim"] = 0
    assert data.lookup("reduce", DTYPE, CALL, named)[0] is None
    named["dim"] = 1
    named["other_input"] = TENSOR
    assert data.lookup("reduce", DTYPE, CALL, named)[0] is None
