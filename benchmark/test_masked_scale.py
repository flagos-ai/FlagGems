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

from . import base, utils

_H800_SHA256 = "3c1a2aa51388456441036b15450002e9e7e936ef405acae3d748c8f30d07c3fa"
_H800_COLLECTION = "335d9a9179c04034d5a1e77c291666857984a922"
_H800_ARTIFACT_COMMIT = "c817c23c999aa22845bc3622d85a2ef68f442336"
_RECIPE_SOURCE = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
_CORE_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr42:op_pytorch_h800_masked_scale_core_baseline": {
        "sha256": "3c1a2aa51388456441036b15450002e9e7e936ef405acae3d748c8f30d07c3fa",
        "artifact": "op_pytorch_h800_masked_scale_core_baseline.json",
        "operators": ["masked_scale"],
        "provenance": {
            "op_pytorch_h800_masked_scale_core_baseline.provenance.json": {
                "key": "op_pytorch_h800_masked_scale_core_baseline.provenance.json",
                "sha256": "71d31ecda6bb0909790b3a10dc72c36982407ae6d749d6d2af94e3105f8dca40",
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


@dataclass
class _MaskedScaleMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _H800Latency:
    @cached_property
    def _h800(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = _aten_read_bytes(
            path, ("pr42:op_pytorch_h800_masked_scale_core_baseline")
        )
        if hashlib.sha256(payload).hexdigest() != _H800_SHA256:
            raise ValueError("Expected the reviewed native H800 masked-scale snapshot")
        data = json.loads(payload)
        if data["masked_scale"]["native_api"] != "ops.masked_scale":
            raise ValueError("Expected native aten::_masked_scale collection")
        return data, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=_H800_SHA256,
            artifact_git_commit=_H800_ARTIFACT_COMMIT,
            collection_source_commit=_H800_COLLECTION,
            ratio="H800 native latency_ms / locally timed target candidate latency_ms",
            input_identity="Original builders, seed0 Gaussian/uint8 draw order and scale2; backend random payloads are not identical.",
        )

    def _import_latency(self, args, kwargs):
        if len(args) != 3 or kwargs:
            raise ValueError("Expected the original input/mask/scale ABI")
        inp, mask, scale = args
        if (
            inp.dtype != torch.float32 or mask.dtype != torch.uint8
            or tuple(inp.shape) not in _CORE_SHAPES or mask.shape != inp.shape
            or mask.device != inp.device or not inp.is_contiguous() or not mask.is_contiguous()
            or inp.requires_grad or mask.requires_grad
            or type(scale) is not float or scale != 2.0
            or inp.untyped_storage().data_ptr() == mask.untyped_storage().data_ptr()
        ):
            raise ValueError("Static snapshot requires the original dense Float32/Uint8/scale2 configuration")
        # Validate recipe values outside both native/candidate timers.
        cpu_input, cpu_mask = inp.detach().cpu(), mask.detach().cpu()
        if not torch.isfinite(cpu_input).all() or not (cpu_mask < 2).all():
            raise ValueError("Static snapshot requires finite Gaussian input and binary uint8 mask")
        def metadata(value):
            return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()), requires_grad=value.requires_grad)
        config = dict(
            inputs=dict(args=[metadata(inp), metadata(mask), scale], kwargs={}),
            source_commit=_RECIPE_SOURCE, benchmark="benchmark/test_masked_scale.py",
            descriptor=list(inp.shape), scope="core", variant=0,
            input_recipe=dict(seed=0, input="torch.randn", mask="torch.randint", mask_low=0, mask_high=2,
                              mask_dtype="torch.uint8", draw_order=["input", "mask"], scale=2.0),
        )
        data, source = self._h800
        encoded = json.dumps(config, sort_keys=True, allow_nan=False)
        rows = [row for group in data["masked_scale"]["shapes"].values()
                for dtype, row in group.items() if dtype == str(inp.dtype)
                and json.dumps(row.get("config"), sort_keys=True, allow_nan=False) == encoded]
        if len(rows) != 1:
            raise ValueError("Expected one native H800 record matching the complete original typed recipe")
        row = rows[0]
        origin = row["collection_origin"]
        environment, task = origin["environment"], origin["task"]
        gpu, run = environment["hardware"]["gpu"], environment["run"]
        if (
            gpu["name"] != "NVIDIA H800"
            or gpu["uuid"].removeprefix("GPU-") != "7d545def-d5ba-4424-b859-2067806337c2"
            or environment["source"]["git_commit"] != _H800_COLLECTION
            or environment["source"]["git_dirty"] is not False
            or not environment["software"]["torch_cuda_version"]
            or run["repo"] != "pytorch" or run["ncu_enabled"] is not False
            or run["selected_ops"] != ["masked_scale"] or task["status"] != "completed"
            or type(task["returncode"]) is not int or task["returncode"] != 0 or task["device"] != "3"
            or row["metric_validity"] != {"T_us": "measured", "latency_ms": "measured"}
        ):
            raise ValueError("Static latency lacks the exact clean native H800 worker provenance")
        for key in ("T_us", "latency_ms"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Native H800 latency must be finite positive measured time")
        if not math.isclose(row["T_us"] / 1000, row["latency_ms"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Native H800 microsecond/millisecond units disagree")
        self._imported_baseline = dict(**source, config=config, collection_origin=origin,
                                      ncu_enabled=False, latency_validity="measured",
                                      target_timing_mode=base.Config.mode.value)
        return row["latency_ms"]

    def get_latency(self, op, *args, **kwargs):
        if getattr(self, "_native_measurement_pending", False):
            if op is not self.torch_op:
                raise ValueError("Static H800 data replaces only the reference timer")
            self._native_measurement_pending = False
            return self._import_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if flag_gems.vendor_name == "enflame" and not external:
            raise ValueError("Enflame masked-scale requires FLAGGEMS_NV_BASELINE native H800 data")
        if external and (self.is_backward or base.Config.skip_native or base.Config.reference_only):
            raise ValueError("Static masked-scale data requires ordinary forward candidate timing")
        self._imported_baseline = None
        self._native_measurement_pending = external and "latency_base" in self.to_bench_metrics
        try:
            metric = super()._measure_input(input, case_id)
            if self._imported_baseline is not None:
                value = metric.latency
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Actual local candidate latency must be positive and finite")
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / value
                return _MaskedScaleMetrics(**asdict(metric), baseline_source=self._imported_baseline)
            return metric
        finally:
            self._native_measurement_pending = False


class MaskedScaleBenchmark(_H800Latency, base.Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(dims), "mask": list(dims)},
                    params={"scale": 2.0},
                    builder_args=(dims,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        torch.manual_seed(0)
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        if flag_gems.vendor_name == "cambricon":
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device="cpu").to(
                self.device
            )
        else:
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=self.device)
        return inp, mask, plan.params["scale"]

    def get_tflops(self, op, *args, **kwargs):
        return args[0].numel()


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
def test_masked_scale():
    bench = MaskedScaleBenchmark(
        op_name="_masked_scale",
        torch_op=torch.ops.aten._masked_scale,
        dtypes=[torch.float32],
    )
    bench.run()
