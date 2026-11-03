import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts

CORE_SHAPES = [
    (2, 2),
    (3, 2),
    (2, 3),
    (4, 4),
    (8, 4),
    (4, 8),
    (256, 2),
    (2, 256),
    (32, 4, 4),
]
COMPREHENSIVE_SHAPES = [
    (1024, 2),
    (2, 1024),
    (256, 4),
    (4, 256),
    (128, 4, 4),
]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr72:op_musa_linalg_current_core_h800_baseline": {
        "sha256": "8ba73fbe61042343df6d84dcebcd9c087a882ed5b678b7867f531100ce2a5c77",
        "artifact": "op_musa_linalg_current_core_h800_baseline.json",
        "operators": [
            "linalg_cond_current_core",
            "linalg_svd_current_core",
            "linalg_svdvals_current_core",
            "pinverse_current_core",
        ],
        "provenance": {},
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


@dataclass
class _PinverseMetrics(consts.BenchmarkMetrics):
    baseline_source: dict = None


class _PinverseH800Reference:
    @cached_property
    def _h800_reference(self):
        path = Path(os.environ["FLAGGEMS_PINVERSE_H800_BASELINE"]).resolve(strict=True)
        payload = _aten_read_bytes(
            path, ("pr72:op_musa_linalg_current_core_h800_baseline")
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "8ba73fbe61042343df6d84dcebcd9c087a882ed5b678b7867f531100ce2a5c77":
            raise ValueError("Expected the qualified native H800 pinverse artifact")
        data = json.loads(payload)
        name = "pinverse_current_core"
        task = data["_collection"]["tasks"][name]
        environment = data["_environments"][str(task["device"])]
        if (
            data[name]["native_api"] != "ops." + name
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "6b674c0daf4ea87fd44dd2ed5fb6436880df1bb2"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
        ):
            raise ValueError(
                "Expected a completed, clean native H800 pinverse collection"
            )
        return data[name]["shapes"], {
            "artifact": str(path),
            "artifact_sha256": digest,
            "operator": name,
            "collection_task": task,
            "environment": environment,
            "environment_qualification": (
                "Shared collection per-device envelope; "
                "not an operator-specific startup receipt"
            ),
            "input_identity": "Same full input contract and seed recipe; not identical cross-device payloads",
            "ratio": "Raw H800 latency / actual local candidate latency",
        }

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_pinverse_native_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._pinverse_native_pending = False
        if op is not self.torch_op:
            raise ValueError(
                "External pinverse latency replaces only its native reference"
            )
        if len(args) != 1 or kwargs != {"rcond": 1e-15}:
            raise ValueError(
                "Pinverse positional and rcond ABI must match the native record"
            )
        inp = args[0]
        if (
            not isinstance(inp, torch.Tensor)
            or tuple(inp.shape) not in CORE_SHAPES
            or inp.dtype != torch.float32
            or not inp.is_contiguous()
            or inp.storage_offset() != 0
            or inp.requires_grad
            or inp.device.type != torch.device(self.device).type
        ):
            raise ValueError(
                "Pinverse tensor contract differs from the native H800 record"
            )
        expected = {
            "inputs": {
                "args": [
                    {
                        "shape": list(inp.shape),
                        "dtype": str(inp.dtype),
                        "stride": list(inp.stride()),
                        "requires_grad": False,
                    }
                ],
                "kwargs": {"rcond": 1e-15},
            },
            "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            "benchmark": "benchmark/test_pinverse.py",
            "scope": "core",
            "descriptor": list(inp.shape),
            "variant": 0,
            "input_recipe": {"seed": 0, "input": "torch.randn"},
        }
        shapes, provenance = self._h800_reference
        rows = [
            r
            for dtypes in shapes.values()
            for dtype, r in dtypes.items()
            if dtype == str(inp.dtype) and r["config"] == expected
        ]
        if len(rows) != 1:
            raise ValueError(
                "Expected one native H800 record for the complete pinverse input"
            )
        record = rows[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 pinverse latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 pinverse latency units disagree")
        self._pinverse_imported_reference = {**provenance, "config": expected}
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        self._pinverse_native_pending = (
            flag_gems.vendor_name == "mthreads" and needs_native
        )
        self._pinverse_imported_reference = None
        try:
            if self._pinverse_native_pending and (
                self.is_backward or "tflops" in self.to_bench_metrics
            ):
                raise ValueError(
                    "The native H800 pinverse artifact covers forward latency only"
                )
            if self._pinverse_native_pending and not os.environ.get(
                "FLAGGEMS_PINVERSE_H800_BASELINE"
            ):
                raise ValueError("Missing native H800 pinverse reference")
            metric = super()._measure_input(input, case_id)
            if self._pinverse_imported_reference is None:
                return metric
            return _PinverseMetrics(
                **asdict(metric), baseline_source=self._pinverse_imported_reference
            )
        finally:
            self._pinverse_native_pending = False


class PinverseBenchmark(_PinverseH800Reference, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()
        self.shape_desc = "batch..., M, N"

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"rcond": 1e-15},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        return matrix, {"rcond": 1e-15}


@pytest.mark.pinverse
def test_pinverse():
    bench = PinverseBenchmark(
        op_name="pinverse",
        torch_op=torch.pinverse,
        dtypes=[torch.float32],
    )
    bench.run()
