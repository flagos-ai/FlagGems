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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr52:ldexp_inplace_current_core_h800": {
        "sha256": "cde9d47c4da7478951adc6f90013e14575dd561409a4b77b5e5160992c15865f",
        "artifact": "artifacts/ldexp_inplace_current_core_h800.json",
        "operators": ["ldexp_inplace_current_core"],
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


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={"exponent_dtype": "int32"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    return inp, exponent


@dataclass
class _LdexpNativeMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _LdexpStaticBaseline:
    """Only this original in-place core imports its fresh native reference timer."""

    @cached_property
    def _native_records(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        raw = _aten_read_bytes(path, ("pr52:ldexp_inplace_current_core_h800"))
        digest = "cde9d47c4da7478951adc6f90013e14575dd561409a4b77b5e5160992c15865f"
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Expected exact reviewed fresh-native ldexp artifact")
        data = json.loads(raw)
        key = "ldexp_inplace_current_core"
        env, method, task = (
            data["_environment"],
            data["_measurement_method"],
            data["_execution"],
        )
        expected_method = (
            "official collector + operator-specific untimed-refresh CUDA-event timing"
        )
        if (
            task["exit"] != 0
            or task["source_before_after"] is not True
            or task["head"] != "0ca0827b1440a2a20b717c844966205fbe2a44b2"
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["hardware"]["gpu"]["uuid"] != "f76856dc-d02f-95a0-dc05-054190d1ea7d"
            or env["source"]["git_commit"] != task["head"]
            or env["source"]["git_dirty"] is not False
            or env["run"]["only"] != [key]
            or env["run"]["selected_ops"] != [key]
            or env["run"]["ncu_enabled"] is not False
            or env["run"]["latency_method"] != expected_method
            or method["name"] != expected_method
            or method["ncu_profiled"] is not False
            or data[key]["native_api"] != "ops." + key
        ):
            raise ValueError("Incomplete exact fresh-native ldexp provenance")
        records = [r for ds in data[key]["shapes"].values() for r in ds.values()]
        if len(records) != 15 or any(
            r["measurement_method"]["name"] != expected_method
            or r["measurement_method"]["driver_sha256"] != method["driver_sha256"]
            for r in records
        ):
            raise ValueError("Incomplete original fresh-native grid or timing method")
        return records, _aten_qualify_metadata(
            dict(
                kind="external_nvidia",
                artifact=str(path),
                artifact_sha256=digest,
                environment=env,
                execution=task,
                native_measurement_method=method,
                ratio="H800 fresh native event latency_ms / target candidate latency_ms",
            ),
            ("pr52:ldexp_inplace_current_core_h800"),
        )

    def _import_latency(self, args, kwargs):
        if (
            kwargs
            or len(args) != 2
            or not all(isinstance(x, torch.Tensor) for x in args)
        ):
            raise ValueError("Expected original ldexp_ positional tensor ABI")
        inp, exponent = args
        shapes = [
            (1073741824,),
            (64, 64),
            (4096, 4096),
            (64, 512, 512),
            (1024, 1024, 1024),
        ]
        if (
            tuple(inp.shape) not in shapes
            or exponent.shape != inp.shape
            or inp.dtype not in [torch.float16, torch.float32, torch.bfloat16]
            or exponent.dtype != torch.int32
            or exponent.device != inp.device
            or inp.device != torch.empty(0, device=self.device).device
            or not inp.is_contiguous()
            or not exponent.is_contiguous()
            or inp.requires_grad
            or exponent.requires_grad
        ):
            raise ValueError("Actual original ldexp_ input metadata differ")
        describe = lambda x: dict(
            shape=list(x.shape),
            dtype=str(x.dtype),
            stride=list(x.stride()),
            requires_grad=False,
        )
        method = (
            "official collector + operator-specific untimed-refresh CUDA-event timing"
        )
        config = dict(
            source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
            benchmark="benchmark/test_ldexp_.py",
            scope="core",
            variant=0,
            descriptor=list(inp.shape),
            native_api="torch.ops.aten.ldexp_.default",
            inputs=dict(args=[describe(inp), describe(exponent)], kwargs={}),
            input_recipe=dict(
                rng="manual_seed(0)",
                self="randn",
                exponent="randint(-8,9,int32)",
                fresh_inputs=True,
            ),
            measurement_method=method,
            mutation="self only",
            output_alias="self",
        )
        records, source = self._native_records
        matches = [r for r in records if r.get("config") == config]
        if len(matches) != 1:
            raise ValueError("Missing unique native fresh-input ldexp case")
        record = matches[0]
        us, ms = record["T_us"], record["latency_ms"]
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in [us, ms]
        ) or not math.isclose(ms, us / 1000, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Invalid native ldexp latency units")
        self._baseline_source = dict(
            source,
            config=config,
            target_candidate=(
                "locally timed candidate; implementation supplied by the runner"
            ),
        )
        if base.Config.reference_only:
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise ValueError("Cannot identify active ldexp reference case")
            active[0]["baseline_source"] = self._baseline_source
        return ms

    def get_latency(self, op, *args, **kwargs):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if getattr(self, "_native_pending", False) or (
            external and base.Config.reference_only
        ):
            self._native_pending = False
            if op is not self.torch_op:
                raise ValueError("Native data replaces only reference timer")
            return self._import_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if needs_native and flag_gems.vendor_name == "enflame" and not external:
            raise ValueError("Missing fresh native H800 ldexp reference")
        if external and needs_native and self.is_backward:
            raise ValueError("Expected native/candidate latency mode")
        self._baseline_source = None
        self._native_pending = external and needs_native
        try:
            metric = super()._measure_input(input, case_id)
            if self._baseline_source is not None:
                return _LdexpNativeMetrics(
                    **asdict(metric), baseline_source=self._baseline_source
                )
            return metric
        finally:
            self._native_pending = False


class LdexpInplaceBenchmark(
    _LdexpStaticBaseline, base.BinaryPointwiseBenchmark, OperatorBenchmark
):
    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.ldexp_
def test_ldexp_():
    bench = LdexpInplaceBenchmark(
        op_name="ldexp_",
        torch_op=torch.ops.aten.ldexp_.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
