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

import pytest
import torch

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import flag_gems


from . import base, consts
from .generated_operator_utils import OperatorBenchmark


@dataclass
class _InplaceMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _InplaceBaseline:
    def build_inputs(self, case):
        # b is part of the original case identity even for addmm_, whose API
        # ignores batch. Keep the actual plan so equal tensor geometries cannot
        # accidentally select the first of two different recorded bindings.
        self._baseline_case = case
        return OperatorBenchmark.build_inputs(self, case)

    def _external_input_config(self, args):
        case = getattr(self, "_baseline_case", None)
        if case is None or len(args) != 3:
            raise ValueError("Inplace baseline requires the original active case plan")
        plan = case.builder_args[0]
        if len(plan.builder_args) != 5:
            raise ValueError("Invalid inplace case descriptor")
        batch, m, n, k, column_major = plan.builder_args
        if type(column_major) is not bool or plan.params != {
            "column_major": column_major
        }:
            raise ValueError("Invalid inplace column-major case identity")
        if not all(
            isinstance(x, torch.Tensor) and x.layout == torch.strided for x in args
        ):
            raise ValueError("Inplace baseline requires three dense tensor arguments")
        if self.op_name == "addmm_":
            expected = [(m, n), (m, k), (k, n)]
        elif self.op_name == "baddbmm_":
            expected = [(batch, m, n), (batch, m, k), (batch, k, n)]
        else:
            raise ValueError("Unknown inplace baseline operator")
        if [tuple(x.shape) for x in args] != expected:
            raise ValueError(
                "Inputs disagree with the original inplace case descriptor"
            )
        return {
            "inputs": {
                "args": [
                    {
                        "kind": "tensor",
                        "shape": list(x.shape),
                        "stride": list(x.stride()),
                        "dtype": str(x.dtype),
                        "requires_grad": x.requires_grad,
                    }
                    for x in args
                ],
                "kwargs": {},
            },
            "source_commit": "59515a3d1e76025b624574f1e0ace759339deeec",
            "benchmark": "benchmark/test_" + self.op_name + ".py",
            "descriptor": [batch, m, n, k],
            "scope": "core",
            "variant": int(column_major),
            "params": {"column_major": column_major},
            "binding_defaults": {"alpha": 1, "beta": 1},
            "effects": {"mutates": ["self"], "return_alias": "self"},
            "fresh_inputs": True,
            "reference_implementation": "pytorch_native_operator",
            "latency_measurement": {
                "kind": "cuda_events_fresh_inputs",
                "warmup_ms": 25,
                "rep_ms": 100,
                "preparation": "detach+clone every input preserving dense strides; synchronize",
                "reset_in_measured_interval": False,
                "cache_flush_bytes": 256000000,
            },
        }

    @cached_property
    def _nvidia_baseline(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate baseline JSON key: " + key)
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError("Non-finite baseline JSON constant: " + value)

        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        sidecar_path = Path(
            os.environ.get(
                "FLAGGEMS_NV_BASELINE_PROVENANCE",
                str(path.with_suffix(".provenance.json")),
            )
        ).resolve(strict=True)
        sidecar_payload = sidecar_path.read_bytes()
        sidecar = json.loads(
            sidecar_payload,
            object_pairs_hook=unique_keys,
            parse_constant=reject_constant,
        )
        key = self.op_name
        if (
            sidecar["status"] != "PASS_VERIFIED_NATIVE_H800_SNAPSHOT"
            or sidecar["artifact_sha256"] != hashlib.sha256(payload).hexdigest()
            or sidecar["collection_source_commit"]
            != "0eb30dbe5c7f3e1f3af31f2867fe877d259680bc"
            or sidecar["source_clean"] is not True
            or sidecar["controller_environment"]["NVIDIA_TF32_OVERRIDE"] != "0"
            or sidecar["operators"][key]["records"] != 30
            or sidecar["operators"][key]["numerical_or_native_abi_controls"] != "PASS"
        ):
            raise ValueError(
                "Inplace H800 artifact requires its verified provenance sidecar"
            )
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        if (
            data[key]["native_api"] != "ops." + key
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["run"]["repo"] != "pytorch"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or commit != sidecar["collection_source_commit"]
            or environment["source"]["git_dirty"] is not False
            or not environment["software"]["packages"]["torch"]
        ):
            raise ValueError(
                "Inplace baseline requires a completed, traceable native H800 run"
            )
        return data[key]["shapes"], {
            "kind": "external_nvidia",
            "operator": key,
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "provenance_artifact": str(sidecar_path),
            "provenance_sha256": hashlib.sha256(sidecar_payload).hexdigest(),
            "collection_task": task,
            "collection_source_commit": sidecar["collection_source_commit"],
            "controller_environment": sidecar["controller_environment"],
            "environment": environment,
            "reference_implementation": "pytorch_native_operator",
            "input_identity": "same distribution and exact original case descriptor/input contract; random payloads are not byte-identical",
            "ratio": "raw H800 latency / target latency; no peak normalization",
            "target_timing_mode": base.Config.mode.value,
        }

    def _import_nvidia_latency(self, args, kwargs):
        if kwargs or self.is_backward or not self.fresh_inputs or not self.is_inplace:
            raise ValueError(
                "Inplace external baseline supports the original positional core ABI only"
            )
        config = self._external_input_config(args)
        shapes, provenance = self._nvidia_baseline
        matches = [
            r
            for dtypes in shapes.values()
            for dtype, r in dtypes.items()
            if dtype == str(args[0].dtype) and r.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError("Expected one H800 record matching actual inplace inputs")
        record = matches[0]
        latency_us = record.get("T_us")
        if (
            isinstance(latency_us, bool)
            or not isinstance(latency_us, (int, float))
            or not math.isfinite(latency_us)
            or latency_us <= 0
        ):
            raise ValueError("H800 T_us must be finite and positive")
        latency_ms = latency_us / 1000
        recorded_ms = record.get("latency_ms", latency_ms)
        if (
            isinstance(recorded_ms, bool)
            or not isinstance(recorded_ms, (int, float))
            or not math.isfinite(recorded_ms)
            or not math.isclose(recorded_ms, latency_ms, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(args[0].device),
                "device_name": base.torch_device_fn.get_device_name(),
                "torch": str(torch.__version__),
            },
        }
        if base.Config.reference_only:
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError(
                    "Cannot identify active inplace external reference case"
                )
            active[0]["baseline_source"] = self._imported_baseline
        return latency_ms

    def get_latency(self, op, *args, **kwargs):
        external = flag_gems.vendor_name in {"thead", "enflame"} and bool(
            os.environ.get("FLAGGEMS_NV_BASELINE")
        )
        if external and (
            base.Config.reference_only
            or getattr(self, "_native_measurement_pending", False)
        ):
            if base.Config.skip_native or op is not self.torch_op:
                raise ValueError(
                    "External baseline requires original reference measurement"
                )
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        if flag_gems.vendor_name in {"thead", "enflame"} and op is self.torch_op and not external:
            raise ValueError(
                "Native inplace BLAS baseline is unavailable or numerically unreliable on this backend; set FLAGGEMS_NV_BASELINE to the H800 artifact"
            )
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        external = flag_gems.vendor_name in {"thead", "enflame"} and bool(
            os.environ.get("FLAGGEMS_NV_BASELINE")
        )
        if external and (
            self.is_backward
            or not self.fresh_inputs
            or not self.is_inplace
            or base.Config.skip_native
        ):
            raise ValueError(
                "Inplace external baseline requires forward latency measurement without skip_native"
            )
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if (
                external
                and metric.latency is not None
                and (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                )
            ):
                raise ValueError("Target latency must be finite and positive")
            if self._imported_baseline is not None:
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _InplaceMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def _run_reference_cases(self, case_ids):
        first_record = len(base.Config.reference_records)
        if (
            flag_gems.vendor_name in {"thead", "enflame"}
            and os.environ.get("FLAGGEMS_NV_BASELINE")
            and base.Config.skip_native
        ):
            raise ValueError("Do not combine an external baseline with skip_native")
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            for record in base.Config.reference_records[first_record:]:
                if record.get("baseline_source") and record.get("status") == "PASSED":
                    record["stage"] = "external_baseline"


class BlasCaseBenchmark(_InplaceBaseline, base.BlasBenchmark, OperatorBenchmark):
    __init__ = OperatorBenchmark.__init__
    get_input_iter = OperatorBenchmark.get_input_iter

    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter


def _case_fn(shape, dtype):
    del dtype
    b, m, n, k = tuple(shape)
    for column_major in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"self": [b, m, n], "batch1": [b, m, k], "batch2": [b, k, n]},
            params={"column_major": column_major},
            builder_args=(b, m, n, k, column_major),
        )


def _build_inputs_fn(plan, dtype, device):
    b, m, n, k, column_major = plan.builder_args
    torch.manual_seed(0)
    batch1 = torch.randn((b, m, k), dtype=dtype, device=device)
    if column_major:
        batch2 = torch.randn((b, n, k), dtype=dtype, device=device).transpose(1, 2)
    else:
        batch2 = torch.randn((b, k, n), dtype=dtype, device=device)
    self = torch.randn((b, m, n), dtype=dtype, device=device)
    return self, batch1, batch2


@pytest.mark.baddbmm_
def test_baddbmm_():
    BlasCaseBenchmark(
        op_name="baddbmm_",
        torch_op=lambda self, batch1, batch2: self.baddbmm_(batch1, batch2),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
