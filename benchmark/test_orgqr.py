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

ORGQR_SHAPES = [(16, 8), (32, 16), (64, 32), (128, 64), (128, 128)]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr77:op_orgqr_current_core_h800_perf_baseline": {
        "sha256": "2d1e2dde335856f8424f59ba759bbd37951e177348b8a465670489e771458f31",
        "artifact": "op_orgqr_current_core_h800_perf_baseline.json",
        "operators": ["orgqr_current_core"],
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


class OrgqrBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ORGQR_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from self._inputs_for_shape(shape, cur_dtype)

    def _inputs_for_shape(self, shape, cur_dtype):
        matrix = torch.randn(shape, dtype=cur_dtype, device=self.device)
        yield torch.geqrf(matrix)


@dataclass
class _OrgqrH800Metrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _OrgqrH800Baseline:
    @cached_property
    def _orgqr_h800_records(self):
        path = Path(os.environ["FLAGGEMS_ORGQR_H800_BASELINE"]).resolve(strict=True)
        payload = _aten_read_bytes(
            path, ("pr77:op_orgqr_current_core_h800_perf_baseline")
        )
        digest = "2d1e2dde335856f8424f59ba759bbd37951e177348b8a465670489e771458f31"
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Expected the reviewed native H800 orgqr core artifact")
        data = json.loads(payload)
        key = "orgqr_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        records = data[key]["shapes"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[key]["native_api"] != "ops." + key
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "76125c6b7fb92a0131b43b01aa4dd520e1268aaf"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data["_collection"]["configuration"]["ncu_enabled"] is not False
            or sum(len(values) for values in records.values()) != 10
        ):
            raise ValueError("Expected all ten traceable native H800 orgqr records")
        return records, _aten_qualify_metadata(
            dict(
                artifact=str(path),
                artifact_sha256=digest,
                operator=key,
                collection_task=task,
                environment=environment,
                input_identity="Same unseeded caller-RNG factory and complete configuration; not cross-device random-byte identity",
                ratio="Raw native H800 latency / locally timed candidate; no scaling",
                timing_qualification="Shared functional collection, mean latency only; no NCU metrics",
            ),
            ("pr77:op_orgqr_current_core_h800_perf_baseline"),
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_orgqr_h800_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._orgqr_h800_pending = False
        if op is not self.torch_op or self.is_backward or len(args) != 2 or kwargs:
            raise ValueError("Static orgqr requires the original forward native call")
        inp, tau = args
        m, n = inp.shape if inp.ndim == 2 else (0, 0)
        if (
            (m, n) not in ORGQR_SHAPES
            or inp.dtype not in (torch.float32, torch.float64)
            or tau.dtype != inp.dtype
            or tau.shape != (n,)
            or tau.device != inp.device
            or inp.device.type != torch.device(self.device).type
            or inp.stride() != (1, m)
            or tau.stride() != (1,)
            or any(value.storage_offset() != 0 or value.requires_grad for value in args)
        ):
            raise ValueError("Runtime orgqr inputs differ from the H800 core contract")
        expected = dict(
            inputs=dict(
                args=[
                    dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        stride=list(value.stride()),
                        requires_grad=False,
                    )
                    for value in args
                ],
                kwargs={},
            ),
            source_commit="9c280800abb30b0fa3e1c073d564fc4b6d97d1cd",
            benchmark="benchmark/test_orgqr.py",
            scope="core",
            descriptor=[m, n],
            variant=0,
            input_recipe=dict(
                matrix="torch.randn(shape,dtype,device)",
                factorization="torch.geqrf(matrix) outside timed call",
                rng_policy="consume caller RNG stream; no per-case seed reset",
                native="torch.orgqr(input,tau)",
            ),
        )
        records, source = self._orgqr_h800_records
        matches = [
            record
            for values in records.values()
            for dtype, record in values.items()
            if dtype == str(inp.dtype)
            and {
                key: value
                for key, value in record["config"].items()
                if key != "producer_sample"
            }
            == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one H800 record for the full orgqr configuration"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 orgqr latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 orgqr latency units disagree")
        self._orgqr_h800_source = dict(source, config=record["config"])
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if flag_gems.vendor_name != "ascend" or not needs_native:
            return super()._measure_input(input, case_id)
        if not os.environ.get("FLAGGEMS_ORGQR_H800_BASELINE"):
            raise ValueError(
                "Ascend native orgqr uses CPU fallback; set FLAGGEMS_ORGQR_H800_BASELINE to the reviewed native H800 core artifact"
            )
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Static orgqr baseline requires the original forward latency measurement"
            )
        self._orgqr_h800_pending = "latency_base" in self.to_bench_metrics
        self._orgqr_h800_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._orgqr_h800_source is not None:
                if metric.latency is not None and (
                    isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target orgqr candidate must have a real positive latency"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _OrgqrH800Metrics(
                    **asdict(metric), baseline_source=self._orgqr_h800_source
                )
            return metric
        finally:
            self._orgqr_h800_pending = False


class _OrgqrCaseBenchmark(_OrgqrH800Baseline, OrgqrBenchmark):
    def _inputs_for_shape(self, shape, cur_dtype):
        if cur_dtype == torch.float64 and flag_gems.vendor_name in (
            "kunlunxin",
            "thead",
        ):
            pytest.skip(
                "Native orgqr reference cannot preserve float64 on this backend"
            )
        if (
            cur_dtype == torch.float64 and flag_gems.vendor_name == "iluvatar"
        ) or flag_gems.vendor_name == "thead":
            input, tau = torch.geqrf(torch.randn(shape, dtype=cur_dtype, device="cpu"))
            yield input.to(self.device), tau.to(self.device)
        else:
            yield from super()._inputs_for_shape(shape, cur_dtype)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return next(self._inputs_for_shape(shape, case.dtype))


@pytest.mark.orgqr
def test_orgqr():
    bench = _OrgqrCaseBenchmark(
        op_name="orgqr",
        torch_op=torch.orgqr,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()


@pytest.mark.orgqr_out
def test_orgqr_out():
    def torch_orgqr_out(input, tau):
        return torch.orgqr(input, tau, out=torch.empty_like(input))

    def gems_orgqr_out(input, tau):
        return flag_gems.orgqr_out(input, tau, out=torch.empty_like(input))

    bench = OrgqrBenchmark(
        op_name="orgqr_out",
        torch_op=torch_orgqr_out,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(gems_orgqr_out)
    bench.run()
