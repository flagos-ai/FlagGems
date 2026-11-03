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

# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr52:kaiser_window_current_core_h800": {
        "sha256": "7cfcf7426f705de7b52371f9ca7818a7201a4a82427cf6e03105d055021cb99e",
        "artifact": "artifacts/kaiser_window_current_core_h800.json",
        "operators": ["kaiser_window_current_core"],
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


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"periodic": True, "beta": 12.0},
        builder_args=(shape, True),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"periodic": False, "beta": 12.0},
            builder_args=(shape, False),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, periodic = plan.builder_args
    return (
        {
            "window_length": math.prod(shape),
            "periodic": periodic,
            "beta": 12.0,
            "dtype": dtype,
            "device": device,
        },
    )


@dataclass
class _KaiserNativeMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _KaiserStaticBenchmark(base.GenericBenchmark):
    """Only the original plain Kaiser core; import the native reference timer."""

    @cached_property
    def _native_records(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        raw = _aten_read_bytes(path, ("pr52:kaiser_window_current_core_h800"))
        digest = "7cfcf7426f705de7b52371f9ca7818a7201a4a82427cf6e03105d055021cb99e"
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Expected exact reviewed native Kaiser artifact")
        data = json.loads(raw)
        key = "kaiser_window_current_core"
        task = data["_collection"]["tasks"][key]
        env = data["_environment"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["hardware"]["gpu"]["uuid"] != "555633f8-c39c-bb8a-d549-791cb8451f44"
            or env["source"]["git_commit"] != "3975e9fa30eb83a5d2889ddc6baaca8fb29406f7"
            or env["source"]["git_dirty"] is not False
            or env["run"]["only"] != [key]
            or env["run"]["selected_ops"] != [key]
            or env["run"]["ncu_enabled"] is not False
            or data[key]["native_api"] != "ops." + key
        ):
            raise ValueError("Incomplete exact native Kaiser provenance")
        records = [r for ds in data[key]["shapes"].values() for r in ds.values()]
        if len(records) != 15:
            raise ValueError("Incomplete original Kaiser grid")
        return records, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=digest,
            environment=env,
            collection_task=task,
            ratio="H800 native latency_ms / target candidate latency_ms",
        )

    def _import_latency(self, args, kwargs):
        if args or set(kwargs) != {
            "window_length",
            "periodic",
            "beta",
            "dtype",
            "device",
        }:
            raise ValueError("Expected original Kaiser factory keyword ABI")
        case_id = getattr(self, "_native_case_id", None)
        if base.Config.reference_only:
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise ValueError("Cannot identify exact original Kaiser case")
            case_id = active[0]["case_id"]
        cases = [c for c in self._collect_cases() if c.case_id == case_id]
        if len(cases) != 1:
            raise ValueError("Missing unique original factory case identity")
        case = cases[0]
        descriptor = list(case.shape["input"])
        shapes = [
            [1073741824],
            [64, 64],
            [4096, 4096],
            [64, 512, 512],
            [1024, 1024, 1024],
        ]
        if (
            descriptor not in shapes
            or case.params != {"periodic": True, "beta": 12.0}
            or kwargs["window_length"] != math.prod(descriptor)
            or type(kwargs["window_length"]) is not int
            or kwargs["periodic"] is not True
            or kwargs["beta"] != 12.0
            or type(kwargs["beta"]) is not float
            or kwargs["dtype"] != case.dtype
            or torch.empty(0, device=kwargs["device"]).device
            != torch.empty(0, device=self.device).device
        ):
            raise ValueError("Actual factory arguments differ from original core")
        config = dict(
            source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
            benchmark="benchmark/test_kaiser_window.py",
            scope="core",
            variant=0,
            descriptor=descriptor,
            native_api="torch.kaiser_window",
            inputs=dict(
                args=[],
                kwargs=dict(
                    window_length=math.prod(descriptor),
                    periodic=True,
                    beta=12.0,
                    dtype=str(case.dtype),
                    device="target",
                ),
            ),
            input_recipe="Exact original _build_inputs_fn; factory has no input tensor/RNG",
            output=dict(
                shape=[math.prod(descriptor)],
                dtype=str(case.dtype),
                device="target",
                alias=False,
                mutation=False,
            ),
        )
        records, source = self._native_records
        matches = [r for r in records if r.get("config") == config]
        if len(matches) != 1:
            raise ValueError("Missing unique native full factory case")
        record = matches[0]
        us, ms = record["T_us"], record["latency_ms"]
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in [us, ms]
        ) or not math.isclose(ms, us / 1000, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Invalid native Kaiser latency units")
        self._baseline_source = dict(
            source,
            config=config,
            case_id=case_id,
            target_candidate=(
                "locally timed candidate; implementation supplied by the runner"
            ),
        )
        if base.Config.reference_only:
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
        if (
            needs_native
            and flag_gems.vendor_name in ("enflame", "ascend", "kunlunxin")
            and not external
        ):
            raise ValueError("Missing native H800 Kaiser reference")
        if external and needs_native and self.is_backward:
            raise ValueError("Expected native/candidate latency mode")
        self._native_case_id = case_id
        self._baseline_source = None
        self._native_pending = external and needs_native
        try:
            metric = super()._measure_input(input, case_id)
            if self._baseline_source is not None:
                return _KaiserNativeMetrics(
                    **asdict(metric), baseline_source=self._baseline_source
                )
            return metric
        finally:
            self._native_pending = False
            self._native_case_id = None


@pytest.mark.kaiser_window
def test_kaiser_window():
    bench = _KaiserStaticBenchmark(
        op_name="kaiser_window",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.kaiser_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.kaiser_window_beta
def test_kaiser_window_beta():
    bench = base.GenericBenchmark(
        op_name="kaiser_window_beta",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.kaiser_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
