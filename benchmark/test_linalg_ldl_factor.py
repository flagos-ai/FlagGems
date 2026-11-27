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
from .generated_operator_utils import OperatorBenchmark

LDL_FACTOR_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr70:op_ldl_factor_current_core_h800_baseline": {
        "sha256": "9a11c5e57869eefdcf01f43889bf340041edd40db78df5dda462565f299792d8",
        "artifact": "op_ldl_factor_current_core_h800_baseline.json",
        "operators": ["linalg_ldl_factor_current_core"],
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
class _LdlFactorMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class LdlFactorBenchmark(OperatorBenchmark):
    @cached_property
    def _h800_records(self):
        path = Path(os.environ["FLAGGEMS_LDL_FACTOR_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = _aten_read_bytes(
            path, ("pr70:op_ldl_factor_current_core_h800_baseline")
        )
        expected_sha = (
            "9a11c5e57869eefdcf01f43889bf340041edd40db78df5dda462565f299792d8"
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError("Expected the reviewed current-core native H800 LDL data")
        data = json.loads(payload)
        key = "linalg_ldl_factor_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[key]["native_api"] != "ops." + key
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "e7178927627a123ec4fcec24cc1cd8d9102b869d"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data["_collection"]["configuration"]["ncu_enabled"] is not False
            or sum(len(ds) for ds in data[key]["shapes"].values()) != 10
        ):
            raise ValueError(
                "Expected the complete ten-record native H800 LDL collection"
            )
        return data[key]["shapes"], _aten_qualify_metadata(
            dict(
                artifact=str(path),
                artifact_sha256=expected_sha,
                operator=key,
                collection_task=task,
                environment=environment,
                input_identity="Exact current configuration/seed recipe; not cross-device random-byte identity",
                ratio="Raw H800 native latency / locally timed target candidate; no scaling",
                timing_qualification="Shared functional collection, latency only; no NCU metrics",
            ),
            ("pr70:op_ldl_factor_current_core_h800_baseline"),
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_ldl_native_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._ldl_native_pending = False
        if op is not self.torch_op:
            raise ValueError("External LDL latency replaces only its native reference")
        if self.is_backward or len(args) != 1 or kwargs != {"hermitian": False}:
            raise ValueError(
                "Static LDL requires the original forward input and explicit hermitian=False"
            )
        (inp,) = args
        if (
            tuple(inp.shape) not in LDL_FACTOR_SHAPES
            or inp.dtype not in (torch.float32, torch.float64)
            or not inp.is_contiguous()
            or inp.storage_offset() != 0
            or inp.requires_grad
            or inp.device.type != torch.device(self.device).type
        ):
            raise ValueError(
                "Runtime LDL input differs from the collected current-core contract"
            )
        config = dict(
            inputs=dict(
                args=[
                    dict(
                        shape=list(inp.shape),
                        dtype=str(inp.dtype),
                        stride=list(inp.stride()),
                        requires_grad=False,
                    )
                ],
                kwargs=kwargs,
            ),
            source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            benchmark="benchmark/test_linalg_ldl_factor.py",
            scope="core",
            descriptor=list(inp.shape),
            variant=0,
            input_recipe=dict(seed=0, input="torch.randn", diagonal_shift="n*eye(n)"),
        )
        records, source = self._h800_records
        matches = [
            r
            for ds in records.values()
            for dtype, r in ds.items()
            if dtype == str(inp.dtype) and r["config"] == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one native H800 LDL record matching the full runtime configuration"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
            or not math.isclose(
                record["latency_ms"], us / 1000, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "Native LDL latency must be positive with consistent microsecond/millisecond units"
            )
        self._imported_baseline = dict(
            **source, config=config, target_timing_mode=base.Config.mode.value
        )
        return us / 1000

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        self._ldl_native_pending = (
            flag_gems.vendor_name in ("mthreads", "ascend") and needs_native
        )
        self._imported_baseline = None
        try:
            if self._ldl_native_pending and (
                self.is_backward or "tflops" in self.to_bench_metrics
            ):
                raise ValueError(
                    "Static H800 LDL supports the original forward latency measurement"
                )
            if self._ldl_native_pending and not os.environ.get(
                "FLAGGEMS_LDL_FACTOR_H800_BASELINE"
            ):
                raise ValueError("Missing native H800 LDL reference")
            metric = super()._measure_input(input, case_id)
            if self._imported_baseline is not None:
                if metric.latency is not None and (
                    not math.isfinite(metric.latency) or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target LDL candidate latency must be positive and finite"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _LdlFactorMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._ldl_native_pending = False

    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_FACTOR_SHAPES
        self.shape_desc = "N, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"hermitian": False},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    if base.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double values")
    (shape,) = plan.builder_args
    if base.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin cannot preserve float64 input storage")
    if base.vendor_name == "iluvatar" and dtype == torch.float64:
        pytest.skip("Iluvatar native ldl_factor does not support float64")
    torch.manual_seed(0)
    n = shape[-1]
    factory_device = (
        "cpu" if base.vendor_name == "ascend" and dtype == torch.float64 else device
    )
    inp = torch.randn(shape, dtype=dtype, device=factory_device)
    inp = inp @ inp.transpose(-2, -1) + n * torch.eye(
        n, dtype=dtype, device=factory_device
    )
    return inp.to(device), {"hermitian": False}


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
def test_linalg_ldl_factor():
    LdlFactorBenchmark(
        op_name="ldl_factor",
        torch_op=torch.linalg.ldl_factor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float64],
    ).run()
