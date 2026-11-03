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

_ACTIVATION_H800_SHA256 = (
    "b4541493b49ece63a686553faa498a966a4376487eabcd9e35bc31b44e513fdd"
)
_ACTIVATION_H800_SOURCE = "04abbf4f2c8378bdc1b034ac9cabc92d7aecb9e7"


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr71:op_activation_current75_h800_baseline": {
        "sha256": "b4541493b49ece63a686553faa498a966a4376487eabcd9e35bc31b44e513fdd",
        "artifact": "op_activation_current75_h800_baseline.json",
        "operators": [
            "addmm_activation_current_core",
            "addmm_activation_out_current_core",
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
class _ActivationMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _ActivationH800Reference:
    """Only Enflame imports exact per-case native CUDA latency records."""

    @cached_property
    def _activation_h800(self):
        path = Path(os.environ["FLAGGEMS_ACTIVATION_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = _aten_read_bytes(path, ("pr71:op_activation_current75_h800_baseline"))
        if hashlib.sha256(payload).hexdigest() != _ACTIVATION_H800_SHA256:
            raise ValueError("Exact reviewed full-core native H800 artifact required")
        data = json.loads(payload)
        records = {}
        for name in (
            "addmm_activation_current_core",
            "addmm_activation_out_current_core",
        ):
            task = data["_collection"]["tasks"][name]
            env = data["_environments"][str(task["device"])]
            if (
                task["status"] != "completed"
                or task["returncode"] != 0
                or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
                or env["source"]["git_commit"] != _ACTIVATION_H800_SOURCE
                or env["source"]["git_dirty"] is not False
                or env["run"]["repo"] != "pytorch"
                or data[name]["native_api"] != "ops." + name
                or data["_collection"]["configuration"]["ncu_enabled"] is not False
            ):
                raise ValueError("Completed clean native CUDA H800 collection required")
            cases = {}
            for shape_key, dtypes in data[name]["shapes"].items():
                for dtype, record in dtypes.items():
                    config = record["config"]
                    cid = config["original_case_id"]
                    if cid in cases or dtype != config["recipe"]["original_dtype"]:
                        raise ValueError("Unique original joint H800 identity required")
                    us = record["T_us"]
                    if (
                        isinstance(us, bool)
                        or not isinstance(us, (int, float))
                        or not math.isfinite(us)
                        or us <= 0
                        or not math.isclose(
                            record["latency_ms"], us / 1000, rel_tol=1e-9, abs_tol=1e-12
                        )
                        or record.get("timing", {}).get("return_mode") != "median"
                    ):
                        raise ValueError(
                            "Finite median H800 latency and consistent units required"
                        )
                    cases[cid] = (record, shape_key)
            if len(cases) != 75:
                raise ValueError("All original 75 H800 records must be present")
            observations = data["_activation_math_flags"]["operators"][name]
            for label in ("startup", "exit"):
                flags = observations[label]
                if (
                    flags["matmul_allow_tf32"] is not True
                    or flags["cudnn_allow_tf32"] is not True
                    or flags["float32_matmul_precision"] != "high"
                ):
                    raise ValueError("Original default native H800 math flags required")
            records[name] = (cases, env, task, observations)
        return path, records

    def build_inputs(self, case):
        self._activation_input_case = case
        return super().build_inputs(case)

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "enflame" and not overridden:
            raise ValueError("Enflame activation requires an explicit candidate")
        return op, dispatch, overridden

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name != "enflame" or op is not self.torch_op:
            return super().get_latency(op, *args, **kwargs)
        if (
            self.is_backward
            or base.Config.skip_native
            or base.Config.base_data is not None
            or base.Config.bench_level != consts.BenchLevel.CORE
            or self.op_name not in ("_addmm_activation", "_addmm_activation_out")
        ):
            raise ValueError(
                "Original forward core and local static H800 reference required"
            )
        case = self._activation_input_case
        module = (
            "addmm_activation"
            + ("_out" if self.op_name.endswith("_out") else "")
            + "_current_core"
        )
        path, modules = self._activation_h800
        records, env, task, observations = modules[module]
        record, key = records[case.case_id]
        config = record["config"]
        if (
            case.shape != config["original_shape"]
            or case.params != config["original_params"]
            or case.ordinal != config["ordinal"]
            or str(case.dtype) != config["recipe"]["original_dtype"]
        ):
            raise ValueError(
                "Original complete case ID/shape/dtype/parameter joint required"
            )
        target = torch.device(self.device)
        if target.index is None:
            target = torch.device(target.type, base.torch_device_fn.current_device())

        def describe(value):
            if not isinstance(value, torch.Tensor):
                return value
            if value.device != target:
                raise ValueError("Actual original target device required")
            return dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                stride=list(value.stride()),
                storage_offset=value.storage_offset(),
                requires_grad=value.requires_grad,
            )

        actual = dict(
            args=[describe(t) for t in args],
            kwargs={k: describe(v) for k, v in kwargs.items()},
        )
        if actual != config["inputs"] or len(args) != 3:
            raise ValueError(
                "Actual call must match every native H800 input and keyword"
            )
        tensors = [*args, *[t for t in kwargs.values() if isinstance(t, torch.Tensor)]]
        if len({t.untyped_storage()._cdata for t in tensors}) != len(tensors):
            raise ValueError(
                "Original independent bias/matrices/output storage required"
            )
        self._activation_imported_source = _aten_qualify_metadata(
            dict(
                artifact=str(path),
                artifact_sha256=_ACTIVATION_H800_SHA256,
                operator=module,
                original_case_id=case.case_id,
                shape_key=key,
                config=config,
                collection_task=task,
                hardware=env["hardware"],
                software=env["software"],
                source=env["source"],
                actual_native_math_flags=observations,
                baseline="literal native CUDA API on H800; not local GCU or CPU timing",
                candidate="locally timed explicit candidate; bootstrap runs may use a CPU oracle",
                input_identity="Exact original recipe/configuration; random cross-machine payload bytes differ",
                timing="raw H800 median / local functional candidate timer; no normalization or isolated performance claim",
                target_timing_mode=base.Config.mode.value,
            ),
            ("pr71:op_activation_current75_h800_baseline"),
        )
        return record["T_us"] / 1000

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name != "enflame":
            return super()._measure_input(input, case_id=case_id)
        if case_id != self._activation_input_case.case_id:
            raise ValueError("Built original case must match the measured case")
        self._activation_imported_source = None
        metric = super()._measure_input(input, case_id=case_id)
        source = self._activation_imported_source
        if source is None or metric.latency_base is None or metric.latency is None:
            raise ValueError(
                "Both exact H800 reference and real local candidate timers required"
            )
        if not math.isfinite(metric.latency) or metric.latency <= 0:
            raise ValueError("Local candidate latency must be finite and positive")
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        if hashlib.sha256(
            _aten_read_bytes(
                Path(source["artifact"]), ("pr71:op_activation_current75_h800_baseline")
            )
        ).hexdigest() != _ACTIVATION_H800_SHA256 or Path(
            os.environ["FLAGGEMS_ACTIVATION_H800_BASELINE"]
        ).resolve() != Path(
            source["artifact"]
        ):
            raise ValueError("Static H800 artifact changed around the candidate timer")
        return _ActivationMetrics(**asdict(metric), baseline_source=source)


class _ActivationBenchmark(_ActivationH800Reference, base.BlasBenchmark):
    pass


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2,


@pytest.mark.addmm_activation
@pytest.mark.underscore_addmm_activation
def test_addmm_activation(monkeypatch):
    bench = _ActivationBenchmark(
        op_name="_addmm_activation",
        input_fn=_input_fn,
        torch_op=torch._addmm_activation,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    out = torch.empty([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2, {"out": out}


@pytest.mark.addmm_activation_out
@pytest.mark.underscore_addmm_activation_out
def test_addmm_activation_out(monkeypatch):
    bench = _ActivationBenchmark(
        op_name="_addmm_activation_out",
        input_fn=_input_fn_out,
        torch_op=torch._addmm_activation,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
