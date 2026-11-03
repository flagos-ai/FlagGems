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

import hashlib
import inspect
import json
import math
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .baseline import describe_call, positive_number

_OPERATOR = "_adaptive_avg_pool3d_backward"
_H800_PRODUCER_COMMIT = "c8c928287c1c01e74601d163e5ba6a9fafb93af6"
_INPUT_BUILDER_SHA256 = (
    "6658ad2c7da71d182bb2a0137ba1de5e0880b12b003e9a8aa5e3afa2a2ace6c1"
)


CASES = [
    ((1, 3, 8, 8, 8), (4, 4, 4)),
    ((2, 3, 16, 16, 16), (8, 8, 8)),
    ((1, 1, 32, 32, 32), (16, 16, 16)),
    ((4, 8, 64, 64, 64), (32, 32, 32)),
]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr47:op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028": {
        "sha256": "8211d460ad8ab12a15443ab4d16bccd0ba044dc6b1c55d746867df6893ee4651",
        "artifact": "op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028.json",
        "operators": ["_adaptive_avg_pool3d_backward"],
        "provenance": {
            "op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028.provenance.json": {
                "key": "op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028.provenance.json",
                "sha256": "b528b893ca9ff427244794e0cda002127024dd68a2f5ed95da09f67bcd0e72c3",
            }
        },
    }
}


def _aten_unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate consolidated baseline key: " + key)
        result[key] = value
    return result


def _aten_load_json(raw):
    def reject(value):
        raise ValueError("Nonfinite consolidated baseline constant: " + value)

    return json.loads(raw, object_pairs_hook=_aten_unique_keys, parse_constant=reject)


def _aten_project_snapshot(data, source_id):
    spec = _ATEN_SNAPSHOT_SPECS[source_id]
    manifest = data["_aten_consolidation"]
    if spec.get("auxiliary"):
        entry = manifest["auxiliary_artifacts"][spec["auxiliary"]]
        raw = entry["raw_json"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Consolidated baseline manifest digest differs")
        if _aten_load_json(raw) != entry["content"]:
            raise ValueError(
                "Consolidated manifest content differs from original bytes"
            )
        return raw, None
    source = manifest["sources"][source_id]
    raw = source["raw_json"].encode("utf-8")
    if (
        hashlib.sha256(raw).hexdigest() != spec["sha256"]
        or source["artifact_sha256"] != spec["sha256"]
    ):
        raise ValueError("Consolidated native snapshot digest differs")
    original = _aten_load_json(raw)
    original_ops = {
        key
        for key, value in original.items()
        if isinstance(value, dict) and "shapes" in value
    }
    if set(source["operators"]) != set(spec["operators"]):
        raise ValueError("Consolidated native operator scope differs")
    if source["original_metadata"] != {
        key: value for key, value in original.items() if key not in original_ops
    }:
        raise ValueError("Consolidated collection metadata differs")
    for operator in spec["operators"]:
        entry = source["operators"][operator]
        expected = original[operator]
        if entry["entry_metadata"] != {
            key: value for key, value in expected.items() if key != "shapes"
        } or set(entry["shape_keys"]) != set(expected["shapes"]):
            raise ValueError("Consolidated native entry or case scope differs")
        seen = set()
        for old_key, new_key in entry["shape_keys"].items():
            if (
                new_key in seen
                or new_key != source_id + ":" + old_key
                or data[operator]["shapes"][new_key] != expected["shapes"][old_key]
            ):
                raise ValueError("Consolidated native timing/input record differs")
            seen.add(new_key)
    return raw, source


@lru_cache(maxsize=4)
def _aten_cached_payload(raw, source_id):
    if source_id is None:
        return raw, None
    data = _aten_load_json(raw)
    if "_aten_consolidation" not in data:
        return raw, None
    return _aten_project_snapshot(data, source_id)


def _aten_existing_path(path, source_id=None):
    path = Path(path)
    spec = _ATEN_SNAPSHOT_SPECS.get(source_id, {})
    original_name = Path(spec.get("artifact", spec.get("auxiliary", ""))).name
    if not path.exists() and path.name == original_name:
        unified = path.parent / "op_pytorch_h800_perf_baseline.json"
        if unified.is_file():
            return unified
    return path


def _aten_read_bytes(path, source_id):
    raw = _aten_existing_path(path, source_id).read_bytes()
    return _aten_cached_payload(raw, source_id)[0]


@lru_cache(maxsize=8)
def _aten_cached_view(baseline, source_id, payload):
    data = _aten_load_json(payload)
    for alias, original in (
        _ATEN_SNAPSHOT_SPECS[source_id].get("operator_aliases", {}).items()
    ):
        data[alias] = data[original]
    return type(baseline)(data, _aten_existing_path(baseline.source, source_id))


def _aten_baseline_view(baseline, source_id):
    if baseline is None or source_id is None:
        return baseline
    if "_aten_consolidation" not in baseline.data:
        aliases = _ATEN_SNAPSHOT_SPECS[source_id].get("operator_aliases", {})
        if not aliases or not all(
            original in baseline.data for original in aliases.values()
        ):
            return baseline
        payload = Path(baseline.source).read_bytes()
        if (
            hashlib.sha256(payload).hexdigest()
            != _ATEN_SNAPSHOT_SPECS[source_id]["sha256"]
        ):
            raise ValueError("Standalone native profile digest differs")
        return _aten_cached_view(baseline, source_id, payload)
    payload, _ = _aten_project_snapshot(baseline.data, source_id)
    return _aten_cached_view(baseline, source_id, payload)


def _aten_capture_context(context, operator, source_id):
    if context is None or source_id is None:
        return context
    spec = _ATEN_SNAPSHOT_SPECS[source_id]
    allowed = spec.get("canonical_operators")
    if allowed is None:
        allowed = tuple(spec.get("operators", ())) + tuple(
            spec.get("operator_aliases", ())
        )
    if allowed and operator not in allowed:
        return context
    return _aten_baseline_view(context[0], source_id), context[1], context[2]


class AdaptiveAvgPool3DBackwardBenchmark(base.Benchmark):
    def _require_external_reference(self, args, kwargs):
        if (
            self.op_name != _OPERATOR
            or not base.Config.skip_native
            or base.Config.base_data is None
        ):
            raise ValueError("CPU-fallback backend requires matching H800 --base-data")
        if (
            len(args) != 2
            or kwargs
            or not all(isinstance(x, torch.Tensor) for x in args)
        ):
            raise ValueError("H800 profile requires the original grad_output/self ABI")
        grad, inp = args
        matching = [size for shape, size in CASES if tuple(inp.shape) == shape]
        if len(matching) != 1 or inp.dtype not in consts.FLOAT_DTYPES:
            raise ValueError("H800 profile has no matching original core descriptor")
        if tuple(grad.shape) != (*inp.shape[:-3], *matching[0]):
            raise ValueError("H800 profile gradient geometry differs")
        for value in args:
            if (
                value.device != inp.device
                or value.dtype != inp.dtype
                or value.layout != torch.strided
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.requires_grad
            ):
                raise ValueError("H800 profile input dtype/device/layout differs")
        if grad.untyped_storage()._cdata == inp.untyped_storage()._cdata:
            raise ValueError("H800 profile requires independent input storage")
        builder = inspect.getsource(type(self).build_inputs).lstrip().rstrip()
        if hashlib.sha256(builder.encode()).hexdigest() != _INPUT_BUILDER_SHA256:
            raise ValueError("Current builder differs from the H800 source recipe")
        call = describe_call(args, kwargs)
        expected = {
            "inputs": call,
            "source_commit": "2ff0931a646b97ec98ebd50642b5889619125a6b",
            "benchmark": "benchmark/test__adaptive_avg_pool3d_backward.py",
            "descriptor": list(inp.shape),
            "scope": "core",
            "variant": 0,
            "native_schema": "aten::_adaptive_avg_pool3d_backward.default",
            "grid_contract": "_adaptive_avg_pool3d_backward.kernelgen-dev.core/v1",
            "input_contract": "adaptive-avg-pool3d-backward-randn-gradient/v1",
            "input_builder_sha256": _INPUT_BUILDER_SHA256,
            "producer_selector": {"OPS_BASELINE_ADAPTIVE_AVG_POOL3D_CURRENT_CORE": "1"},
            "random_policy": "source_unseeded_global_generator_no_reset",
            "input_recipe": "randn(self); randn(grad_output); no forward or seed reset",
        }
        baseline = _aten_baseline_view(
            base.Config.base_data,
            ("pr47:op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028"),
        )
        payload = _aten_read_bytes(
            Path(baseline.source),
            ("pr47:op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028"),
        )

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate baseline JSON key")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError("Non-finite baseline JSON constant: " + value)

        canonical = lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        if canonical(data) != canonical(baseline.data):
            raise ValueError("Loaded baseline differs from its source artifact")
        task = data["_collection"]["tasks"][_OPERATOR]
        environment = data["_environments"][str(task["device"])]
        if (
            data[_OPERATOR]["native_api"] != "ops." + _OPERATOR
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _H800_PRODUCER_COMMIT
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["repo"] != "pytorch"
            or not isinstance(environment["software"]["packages"]["torch"], str)
            or not environment["software"]["packages"]["torch"]
        ):
            raise ValueError("A completed clean native H800 profile is required")
        record, key, reason = baseline.lookup(
            _OPERATOR, inp.dtype, call, target_device=self.device
        )
        if (
            record is None
            or reason is not None
            or canonical(record["config"]) != canonical(expected)
        ):
            raise ValueError("H800 input recipe differs or its record is ambiguous")
        latency_us, latency_ms = record.get("T_us"), record.get("latency_ms")
        if (
            not positive_number(latency_us)
            or not positive_number(latency_ms)
            or not math.isclose(
                latency_ms, latency_us / 1000, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError("H800 latency must be positive with matching units")
        return (
            baseline,
            canonical(call),
            latency_ms,
            key,
            hashlib.sha256(payload).hexdigest(),
        )

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name == "enflame":
            self._checked_external_reference = self._require_external_reference(
                *self.unpack_to_args_kwargs(input)
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if flag_gems.vendor_name != "enflame":
            return super()._apply_external_baseline(metric, dtype, context)
        try:
            if context is None or context[1] is None:
                raise ValueError("External reference context is missing")
            baseline, call, named_call = context
            checked, captured, latency, key, artifact = self._checked_external_reference
            if (
                baseline is not checked
                or json.dumps(call, sort_keys=True, separators=(",", ":")) != captured
                or hashlib.sha256(
                    _aten_read_bytes(
                        Path(baseline.source),
                        (
                            "pr47:op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028"
                        ),
                    )
                ).hexdigest()
                != artifact
            ):
                raise ValueError("External reference changed around candidate timing")
            try:
                chip = base.torch_device_fn.get_device_name()
            except (AttributeError, RuntimeError, TypeError):
                chip = ""
            baseline.apply(
                metric,
                _OPERATOR,
                dtype,
                call,
                named_call,
                flag_gems.vendor_name,
                chip,
                target_device=self.device,
            )
            factor = metric.base_scaling_factor
            if (
                metric.error_msg is not None
                or metric.latency_base is not None
                or metric.base_miss_reason is not None
                or metric.base_data_source != baseline.source
                or metric.base_shape_key != key
                or metric.base_latency_ms != latency
                or not all(
                    positive_number(x)
                    for x in (
                        metric.latency,
                        metric.base_raw_speedup,
                        metric.compared_speedup,
                    )
                )
                or (factor is not None and not positive_number(factor))
                or not math.isclose(
                    metric.base_raw_speedup, latency / metric.latency, rel_tol=1e-9
                )
                or not math.isclose(
                    metric.compared_speedup,
                    metric.base_raw_speedup / (factor or 1.0),
                    rel_tol=1e-9,
                )
            ):
                raise ValueError("Candidate timing or H800 comparison is incomplete")
        finally:
            self._checked_external_reference = None

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES
        self.shape_desc = "input shape, output size"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, (shape, output_size) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "self": list(shape),
                        "grad_output": [*shape[:-3], *output_size],
                    },
                    params={},
                    builder_args=(tuple(shape), tuple(output_size)),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, output_size = plan.builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        grad_output = torch.randn(
            (*shape[:-3], *output_size), dtype=case.dtype, device=self.device
        )
        return grad_output, inp

    def _capture_external_baseline(self, args, kwargs):
        context = super()._capture_external_baseline(args, kwargs)
        return _aten_capture_context(
            context,
            self.op_name,
            ("pr47:op_pytorch_h800_adaptive_avg_pool3d_backward_core_20261028"),
        )


@pytest.mark.underscore_adaptive_avg_pool3d_backward
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="native backward dispatch uses CPU fallback; compare matching H800 --base-data",
)
def test_adaptive_avg_pool3d_backward():
    AdaptiveAvgPool3DBackwardBenchmark(
        op_name="_adaptive_avg_pool3d_backward",
        torch_op=torch.ops.aten._adaptive_avg_pool3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
