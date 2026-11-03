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

from . import base

CHOLESKY_SOLVE_SHAPES = [(4, 1), (8, 2), (16, 4), (32, 4), (64, 8)]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr78:op_cholesky_solve_helper_rowmajor_h800_baseline": {
        "sha256": "0bfceeb8f17e7ccf1f8ae89e458177840e0784036a4a128a896cd2482fb772c3",
        "artifact": "op_cholesky_solve_helper_rowmajor_h800_baseline.json",
        "operators": ["cholesky_solve_helper_rowmajor_core"],
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


@lru_cache(maxsize=1)
def _aten_container_sha(raw):
    return (
        hashlib.sha256(raw).hexdigest()
        if "_aten_consolidation" in _aten_load_json(raw)
        else None
    )


def _aten_qualify_metadata(metadata, source_id):
    if source_id is None:
        return metadata
    path = _aten_existing_path(metadata["artifact"], source_id)
    container_sha = _aten_container_sha(path.read_bytes())
    if container_sha is None:
        return metadata
    selected_sha = _ATEN_SNAPSHOT_SPECS[source_id]["sha256"]
    reported_sha = metadata["artifact_sha256"]
    if reported_sha not in (selected_sha, container_sha):
        raise ValueError("Reported native baseline digest has no matching scope")
    return {
        **metadata,
        "artifact_sha256_scope": (
            "container_file" if reported_sha == container_sha else "selected_snapshot"
        ),
        "snapshot_source_id": source_id,
        "snapshot_sha256": selected_sha,
        "container_sha256": container_sha,
    }


@dataclass
class _CholeskyH800Metrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _CholeskyH800Baseline:
    @cached_property
    def _cholesky_h800_records(self):
        path = Path(os.environ["FLAGGEMS_CHOLESKY_SOLVE_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = _aten_read_bytes(
            path, ("pr78:op_cholesky_solve_helper_rowmajor_h800_baseline")
        )
        digest = "0bfceeb8f17e7ccf1f8ae89e458177840e0784036a4a128a896cd2482fb772c3"
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(
                "Expected the published native H800 Cholesky solve artifact"
            )
        data = json.loads(payload)
        task = data["_collection"]["tasks"]["cholesky_solve_helper_rowmajor_core"]
        environment = data["_environments"][str(task["device"])]
        operator = data["cholesky_solve_helper_rowmajor_core"]
        records = operator["shapes"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or operator["native_api"] != "ops.cholesky_solve_helper_rowmajor_core"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "8d6871d2b07cc52296d905905ed8ac1d284e6c4c"
            or environment["source"]["git_dirty"] is not False
            or sum(len(values) for values in records.values()) != 5
        ):
            raise ValueError(
                "Expected five traceable native H800 Cholesky solve records"
            )
        return records, _aten_qualify_metadata(
            dict(
                artifact=str(path),
                artifact_sha256=digest,
                operator="cholesky_solve_helper_rowmajor_core",
                collection_task=task,
                environment=environment,
                input_identity="Same matrix construction and full input contract; not cross-device random-byte identity",
                ratio="Raw native H800 latency / locally timed candidate; no scaling",
            ),
            ("pr78:op_cholesky_solve_helper_rowmajor_h800_baseline"),
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_cholesky_h800_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._cholesky_h800_pending = False
        if op is not self.torch_op or self.is_backward or len(args) != 3 or kwargs:
            raise ValueError("Static Cholesky solve requires the original forward call")
        rhs, factor, upper = args
        if (
            type(upper) is not bool
            or upper
            or rhs.ndim != 2
            or (rhs.shape[0], rhs.shape[1]) not in CHOLESKY_SOLVE_SHAPES
            or factor.shape != (rhs.shape[0], rhs.shape[0])
            or rhs.dtype != torch.float32
            or factor.dtype != rhs.dtype
            or factor.device != rhs.device
            or rhs.device.type != torch.device(self.device).type
            or any(
                value.storage_offset() != 0 or value.requires_grad for value in args[:2]
            )
        ):
            raise ValueError(
                "Runtime inputs differ from the H800 Cholesky solve contract"
            )
        expected = dict(
            native_expression="torch.ops.aten._cholesky_solve_helper",
            inputs=dict(
                args=[
                    dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        stride=list(value.stride()),
                        requires_grad=False,
                    )
                    for value in args[:2]
                ]
                + [False],
                kwargs={},
            ),
            source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            benchmark="benchmark/test_cholesky_solve_helper.py",
            descriptor=list(rhs.shape),
            scope="core",
            variant=0,
            input_recipe={
                "matrix": "randn((n,n),dtype,device) @ matrix.mT + n*eye(n)",
                "factor": (
                    "linalg.cholesky outside timed call; "
                    "row-major factor matches actual original Ascend layout"
                ),
                "rhs": "randn((n,nrhs),dtype,device)",
                "seed": "original torch.manual_seed(0) per case",
            },
        )
        records, source = self._cholesky_h800_records
        matches = [
            record
            for values in records.values()
            for dtype, record in values.items()
            if dtype == str(rhs.dtype)
            and {
                key: value
                for key, value in record["config"].items()
                if key != "producer_sample"
            }
            == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one full-configuration H800 Cholesky solve record"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._cholesky_h800_source = dict(source, config=record["config"])
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if flag_gems.vendor_name != "ascend" or not needs_native:
            return super()._measure_input(input, case_id)
        if not os.environ.get("FLAGGEMS_CHOLESKY_SOLVE_H800_BASELINE"):
            raise ValueError(
                "Ascend Cholesky solve uses CPU fallback; set "
                "FLAGGEMS_CHOLESKY_SOLVE_H800_BASELINE to the published H800 artifact"
            )
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Static baseline requires the original forward latency measurement"
            )
        self._cholesky_h800_pending = True
        self._cholesky_h800_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._cholesky_h800_source is not None:
                if metric.latency is not None and (
                    isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Candidate must have a real positive latency")
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _CholeskyH800Metrics(
                    **asdict(metric), baseline_source=self._cholesky_h800_source
                )
            return metric
        finally:
            self._cholesky_h800_pending = False


class CholeskySolveHelperBenchmark(_CholeskyH800Baseline, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_SHAPES
        self.shape_desc = "N, NRHS"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            n, nrhs = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"rhs": [n, nrhs], "factor": [n, n]},
                    params={"upper": False},
                    builder_args=(n, nrhs),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        n, nrhs = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn((n, n), dtype=case.dtype, device=self.device)
        matrix = matrix @ matrix.mT + n * torch.eye(
            n, dtype=case.dtype, device=self.device
        )
        factor = torch.linalg.cholesky(matrix)
        rhs = torch.randn((n, nrhs), dtype=case.dtype, device=self.device)
        return rhs, factor, False


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
def test_cholesky_solve_helper():
    bench = CholeskySolveHelperBenchmark(
        op_name="_cholesky_solve_helper",
        torch_op=torch.ops.aten._cholesky_solve_helper,
        dtypes=[torch.float32],
    )
    bench.run()
