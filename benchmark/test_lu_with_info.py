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

_TEST_DTYPES = (
    [torch.float32, torch.float64]
    if flag_gems.vendor_name == "nvidia"
    else [torch.float32]
)


_H800_COLLECTION_SOURCE = "95847c9f7040152cb774fe56183eb280c5764c97"
_H800_ARTIFACT_URL = "https://raw.githubusercontent.com/zhaojs12345/ops-baseline/10a2775cd3674d2029cfeea37cc16b0bf3970949/op_pytorch_h800_lu_with_info_current_core_20261028.json"
_H800_PROVENANCE_URL = "https://raw.githubusercontent.com/zhaojs12345/ops-baseline/10a2775cd3674d2029cfeea37cc16b0bf3970949/op_pytorch_h800_lu_with_info_current_core_20261028.provenance.json"
_H800_ARTIFACT_SHA256 = (
    "ca1f3375753e7397acf29b944f85a8443ebb50a1956e924d81f989057f5e5923"
)
_H800_PROVENANCE_SHA256 = (
    "bf68536d95d3d96ae1b2da12239d434b756884d3a80bcc60cd19cad1e31c5567"
)


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr55:op_pytorch_h800_lu_with_info_current_core_20261028": {
        "sha256": "ca1f3375753e7397acf29b944f85a8443ebb50a1956e924d81f989057f5e5923",
        "artifact": "op_pytorch_h800_lu_with_info_current_core_20261028.json",
        "operators": ["lu_with_info_current_core"],
        "provenance": {
            "op_pytorch_h800_lu_with_info_current_core_20261028.provenance.json": {
                "key": "op_pytorch_h800_lu_with_info_current_core_20261028.provenance.json",
                "sha256": "bf68536d95d3d96ae1b2da12239d434b756884d3a80bcc60cd19cad1e31c5567",
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


def _aten_provenance_bytes(path, source_id, sidecar_path, *, explicit=False):
    raw = _aten_existing_path(path, source_id).read_bytes()
    _, source = _aten_cached_payload(raw, source_id)
    if source is None or explicit or Path(sidecar_path).exists():
        return Path(sidecar_path).read_bytes()
    name = Path(sidecar_path).name
    specs = _ATEN_SNAPSHOT_SPECS[source_id]["provenance"]
    if name not in specs:
        name = (
            Path(_ATEN_SNAPSHOT_SPECS[source_id]["artifact"])
            .with_suffix(".provenance.json")
            .name
        )
    if name not in specs:
        raise ValueError("No pinned provenance for this native snapshot")
    entry = source["adjacent_provenance"][specs[name]["key"]]
    payload = entry["raw_json"].encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != specs[name]["sha256"]:
        raise ValueError("Consolidated native provenance digest differs")
    return payload


@dataclass
class _LuMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class LuWithInfoBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (8, 128, 128),
    ]

    @cached_property
    def _h800_reference(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = _aten_read_bytes(
            path, ("pr55:op_pytorch_h800_lu_with_info_current_core_20261028")
        )
        sidecar_path = Path(
            os.environ.get(
                "FLAGGEMS_NV_BASELINE_PROVENANCE",
                str(path.with_suffix(".provenance.json")),
            )
        ).resolve(strict=False)
        sidecar_payload = _aten_provenance_bytes(
            path,
            ("pr55:op_pytorch_h800_lu_with_info_current_core_20261028"),
            sidecar_path,
            explicit=bool(os.environ.get("FLAGGEMS_NV_BASELINE_PROVENANCE")),
        )

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate LU baseline key: " + key)
                result[key] = value
            return result

        def reject(value):
            raise ValueError("Non-finite LU baseline value: " + value)

        data = json.loads(payload, object_pairs_hook=unique, parse_constant=reject)
        proof = json.loads(
            sidecar_payload, object_pairs_hook=unique, parse_constant=reject
        )
        key = "lu_with_info_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            hashlib.sha256(payload).hexdigest() != _H800_ARTIFACT_SHA256
            or hashlib.sha256(sidecar_payload).hexdigest() != _H800_PROVENANCE_SHA256
            or proof["status"] != "PASS_VERIFIED_NATIVE_H800_LU_CORE_SNAPSHOT"
            or proof["artifact_sha256"] != hashlib.sha256(payload).hexdigest()
            or proof["collection_source_commit"] != _H800_COLLECTION_SOURCE
            or proof["source_clean"] is not True
            or proof["original_recipe_controls"] != 5
            or proof["native_cpu_reconstruction_controls"] != 5
            or data[key]["native_api"] != "ops.lu_with_info_current_core"
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _H800_COLLECTION_SOURCE
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["repo"] != "pytorch"
            or not environment["software"]["packages"]["torch"]
        ):
            raise ValueError("LU requires its verified clean native H800 snapshot")
        records = [
            row
            for values in data[key]["shapes"].values()
            for dtype, row in values.items()
            if dtype == "torch.float32"
        ]
        if len(records) != 5:
            raise ValueError("LU baseline must contain all five original FP32 cases")
        expected = [self._reference_config(shape) for shape in self.DEFAULT_SHAPES]
        if sorted(
            json.dumps(row["config"], sort_keys=True) for row in records
        ) != sorted(json.dumps(config, sort_keys=True) for config in expected):
            raise ValueError("LU baseline differs from the original core input recipe")
        return records, {
            "kind": "external_nvidia",
            "operator": "_lu_with_info",
            "artifact_url": _H800_ARTIFACT_URL,
            "provenance_url": _H800_PROVENANCE_URL,
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "provenance_sha256": hashlib.sha256(sidecar_payload).hexdigest(),
            "collection_source_commit": _H800_COLLECTION_SOURCE,
            "environment": environment,
            "ratio": "raw H800 latency / target latency; no peak normalization",
            "input_identity": "exact original shape/dtype/stride/seed recipe and kwargs; random payloads are not byte-identical across chips",
        }

    @staticmethod
    def _reference_config(shape):
        stride, product = [], 1
        for size in reversed(shape):
            stride.insert(0, product)
            product *= size
        return {
            "operator": "_lu_with_info", "native_api": "torch._lu_with_info",
            "source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0",
            "benchmark": "benchmark/test_lu_with_info.py", "scope": "core",
            "shape": list(shape), "dtype": "torch.float32",
            "inputs": {
                "args": [{"kind": "tensor", "shape": list(shape), "stride": stride,
                          "dtype": "torch.float32", "requires_grad": False}],
                "kwargs": {"pivot": True, "check_errors": False},
            },
            "input_recipe": {"manual_seed": 0, "factory": "torch.randn",
                             "cross_device_rng": "same distribution, not byte-identical"},
            "effects": {"mutates": [], "return_aliases_input": False},
            "reference_implementation": "pytorch_native_operator",
        }

    def get_latency(self, op, *args, **kwargs):
        external_reference = (
            flag_gems.vendor_name == "enflame"
            and op is self.torch_op
            and (
                getattr(self, "_native_pending", False)
                or (getattr(self, "_reference_traversal", False) and base.Config.reference_only)
            )
        )
        if external_reference:
            # Consume before loading, including failures; candidate identity may alias Torch.
            self._native_pending = False
            if self.is_backward or base.Config.skip_native or len(args) != 1:
                raise ValueError("External LU reference requires the original forward ABI")
            value = args[0]
            config = self._reference_config(tuple(value.shape))
            expected = config["inputs"]["args"][0]
            if (
                value.dtype != torch.float32 or value.requires_grad or value.device.type != "gcu"
                or list(value.stride()) != expected["stride"]
                or kwargs != config["inputs"]["kwargs"]
            ):
                raise ValueError("LU inputs differ from the original FP32 core contract")
            records, provenance = self._h800_reference
            matches = [row for row in records if row["config"] == config]
            if len(matches) != 1:
                raise ValueError("Expected one exact native H800 LU record")
            row = matches[0]
            latency = row["T_us"]
            if isinstance(latency, bool) or not isinstance(latency, (float, int)) or not math.isfinite(latency) or latency <= 0:
                raise ValueError("LU H800 latency must be finite and positive")
            milliseconds = latency / 1000
            recorded = row.get("latency_ms", milliseconds)
            if isinstance(recorded, bool) or not isinstance(recorded, (float, int)) or not math.isfinite(recorded) or not math.isclose(recorded, milliseconds, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("LU H800 latency units disagree")
            self._imported_reference = dict(provenance, config=config,
                target_environment={"vendor": flag_gems.vendor_name,
                                    "device": str(value.device), "torch": str(torch.__version__)})
            if getattr(self, "_reference_traversal", False):
                active = [row for row in base.Config.reference_records
                          if row.get("operator") == self.op_name
                          and row.get("status") == "FAILED"
                          and row.get("stage") == "benchmark_reference"]
                if len(active) != 1:
                    raise RuntimeError("Cannot identify active LU reference case")
                active[0]["baseline_source"] = self._imported_reference
            return milliseconds
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._imported_reference = None
        self._native_pending = (
            flag_gems.vendor_name == "enflame"
            and "latency_base" in self.to_bench_metrics
            and not base.Config.skip_native
        )
        try:
            metric = super()._measure_input(input, case_id)
            if self._imported_reference is not None:
                return _LuMetrics(**asdict(metric), baseline_source=self._imported_reference)
            return metric
        finally:
            self._native_pending = False

    def _run_reference_cases(self, case_ids):
        first = len(base.Config.reference_records)
        self._reference_traversal = True
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            self._reference_traversal = False
            self._native_pending = False
            for row in base.Config.reference_records[first:]:
                if row.get("baseline_source") and row.get("status") == "PASSED":
                    row["stage"] = "external_baseline"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = "input shape"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"self": list(shape)},
                    params={"pivot": True, "check_errors": False},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, {"pivot": True, "check_errors": False}


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
def test_lu_with_info():
    LuWithInfoBenchmark(
        op_name="_lu_with_info",
        torch_op=torch._lu_with_info,
        dtypes=_TEST_DTYPES,
    ).run()
