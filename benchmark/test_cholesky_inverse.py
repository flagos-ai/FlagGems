import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base
from .baseline import describe_call, positive_number

_H800_PRODUCER_COMMIT = "a6646f15023574ecd6d29e464d316f2851651252"
_RECIPE_SOURCE_COMMIT = "0b3c96ae2624ed17f85c1499f4ecf64899ec3dcb"
_INPUT_BUILDER_SHA256 = (
    "65ab582417c249d275c752645cf920ab432cd64eff1ef18a55022659dc6f2242"
)

CHOLESKY_INVERSE_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr36:op_pytorch_h800_cholesky_inverse_ascend_core_20261027": {
        "sha256": "cd3f94a27557f672194b989ad53211a81a923c572458200fa0048d5e8e7a32dd",
        "artifact": "op_pytorch_h800_cholesky_inverse_ascend_core_20261027.json",
        "operators": ["cholesky_inverse"],
        "provenance": {
            "op_pytorch_h800_cholesky_inverse_ascend_core_20261027.provenance.json": {
                "key": "op_pytorch_h800_cholesky_inverse_ascend_core_20261027.provenance.json",
                "sha256": "577ab0c07a66fdee660add8d84910c3a650c22674ee6cc42dd026e998c1ec138",
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


class CholeskyInverseBenchmark(base.Benchmark):
    def _require_external_reference(self, dtype, call, named_call=None):
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError("Ascend requires a matching H800 --base-data reference")
        baseline = _aten_baseline_view(
            base.Config.base_data,
            ("pr36:op_pytorch_h800_cholesky_inverse_ascend_core_20261027"),
        )

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate baseline JSON key")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite baseline constant: {value}")

        data = json.loads(
            _aten_read_bytes(
                Path(baseline.source),
                ("pr36:op_pytorch_h800_cholesky_inverse_ascend_core_20261027"),
            ),
            object_pairs_hook=unique_keys,
            parse_constant=reject_constant,
        )
        if data != baseline.data:
            raise ValueError("Loaded baseline differs from its source artifact")
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        if (
            data[self.op_name]["native_api"] != "ops.cholesky_inverse"
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
            raise ValueError("A completed clean native H800 collection is required")
        record, _, reason = baseline.lookup(
            self.op_name, dtype, call, named_call, target_device=self.device
        )
        if record is None or reason is not None:
            raise ValueError("H800 baseline does not match the actual input")
        config = record["config"]
        required = {
            "benchmark": "benchmark/test_cholesky_inverse.py",
            "scope": "core",
            "input_contract": "cholesky-inverse-lower-spd-factor/v1",
            "grid_contract": "cholesky_inverse.kernelgen-dev.core/v1",
            "layout_profile": "ascend-native-factor-row-major/v1",
            "producer_selector": {"OPS_BASELINE_CHOLESKY_ASCEND_CORE": "1"},
            "input_recipe": "manual_seed(0); matrix@matrix.T+n*eye; cholesky; row-major factor; upper=False",
            "input_builder_hash_method": "ast.get_source_segment(build_inputs)",
            "input_builder_sha256": _INPUT_BUILDER_SHA256,
            "flag_gems_source_commit": _RECIPE_SOURCE_COMMIT,
        }
        if (
            dtype != torch.float32
            or config["inputs"] != call
            or set(config) != set(required) | {"inputs"}
            or any(config.get(key) != value for key, value in required.items())
        ):
            raise ValueError("H800 baseline recipe or layout differs")
        factor = call["args"][0]
        shape = factor["shape"]
        if shape not in [list(value) for value in CHOLESKY_INVERSE_SHAPES] or call != {
            "args": [
                {
                    "shape": shape,
                    "dtype": "torch.float32",
                    "stride": [shape[0], 1],
                    "requires_grad": False,
                }
            ],
            "kwargs": {"upper": False},
        }:
            raise ValueError("Only the original Ascend core input ABI is supported")
        latency_us = record.get("T_us")
        latency_ms = record.get("latency_ms")
        if (
            not positive_number(latency_us)
            or not positive_number(latency_ms)
            or not math.isclose(latency_ms, latency_us / 1000, rel_tol=1e-9)
        ):
            raise ValueError("H800 latency must be positive with matching units")
        return latency_ms

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name == "ascend":
            args, kwargs = self.unpack_to_args_kwargs(input)
            self._require_external_reference(args[0].dtype, describe_call(args, kwargs))
        # The shared runner owns the real candidate timer, including when the
        # candidate and torch_op are the same callable. Static data is reference.
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        super()._apply_external_baseline(metric, dtype, context)
        if flag_gems.vendor_name != "ascend":
            return
        if context is None:
            raise ValueError("Ascend external reference context is missing")
        baseline, call, named_call = context
        latency_ms = self._require_external_reference(dtype, call, named_call)
        if (
            metric.error_msg is not None
            or metric.base_miss_reason is not None
            or metric.base_data_source != baseline.source
            or metric.base_latency_ms != latency_ms
            or not all(
                positive_number(value)
                for value in (
                    metric.latency,
                    metric.base_raw_speedup,
                    metric.compared_speedup,
                )
            )
        ):
            raise ValueError("Candidate timing or external comparison is incomplete")
        factor = metric.base_scaling_factor
        if (
            factor is not None
            and not positive_number(factor)
            or not math.isclose(
                metric.base_raw_speedup, latency_ms / metric.latency, rel_tol=1e-9
            )
            or not math.isclose(
                metric.compared_speedup,
                metric.base_raw_speedup / (factor or 1.0),
                rel_tol=1e-9,
            )
        ):
            raise ValueError("External comparison ratios do not match the timings")

    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_INVERSE_SHAPES
        self.shape_desc = "N, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"factor": list(shape)},
                    params={"upper": False},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        if (
            case.dtype == torch.float64
            and torch.device(self.device).type != "cpu"
            and flag_gems.vendor_name in {"ascend", "iluvatar"}
        ):
            pytest.skip(
                "backend float64 matrix arithmetic/Cholesky cannot construct a valid factor"
            )
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        n = shape[-1]
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        positive_definite = (
            matrix @ matrix.transpose(-2, -1)
            + torch.eye(n, dtype=case.dtype, device=self.device) * n
        )
        factor = torch.linalg.cholesky(positive_definite)
        return factor, {"upper": False}

    def _capture_external_baseline(self, args, kwargs):
        context = super()._capture_external_baseline(args, kwargs)
        return _aten_capture_context(
            context,
            self.op_name,
            ("pr36:op_pytorch_h800_cholesky_inverse_ascend_core_20261027"),
        )


@pytest.mark.cholesky_inverse
@pytest.mark.skip_native(
    vendors=["ascend"],
    reason="native cholesky_inverse falls back to CPU; use a matching H800 --base-data reference",
)
def test_cholesky_inverse():
    CholeskyInverseBenchmark(
        op_name="cholesky_inverse",
        torch_op=torch.cholesky_inverse,
        dtypes=[torch.float32, torch.float64],
    ).run()
