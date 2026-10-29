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
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_INPUT_CONTRACT = "cudnn-batch-norm-backward-training-invstd/v1"


@dataclass
class _CrossMachineMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


SHAPES = [(16, 16, 64), (32, 64, 56, 56)]


class OperatorCaseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "operator-specific shape"

    def _external_input_config(self, args):
        if len(args) != 9 or args[3] is not None or args[4] is not None:
            raise ValueError("External cuDNN baseline requires training input ABI")
        inp, grad, weight, _, _, mean, invstd, eps, reserve = args
        tensors = [inp, grad, weight, mean, invstd, reserve]
        if any(t.layout != torch.strided or t.device != inp.device for t in tensors):
            raise ValueError("External cuDNN baseline requires colocated dense tensors")
        if (
            grad.shape != inp.shape
            or grad.dtype != inp.dtype
            or inp.dtype not in (torch.float16, torch.float32)
            or any(
                t.dtype != torch.float32 or t.shape != (inp.shape[1],)
                for t in (weight, mean, invstd)
            )
            or reserve.dtype != torch.uint8
            or reserve.numel() != 0
            or type(eps) is not float
            or eps != 1e-5
        ):
            raise ValueError("External cuDNN baseline input contract differs")

        def encode(value):
            if not isinstance(value, torch.Tensor):
                return value
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "stride": list(value.stride()),
                "requires_grad": value.requires_grad,
            }

        return {
            "inputs": {"args": [encode(value) for value in args], "kwargs": {}},
            "benchmark": "benchmark/test_cudnn_batch_norm_backward.py",
            "scope": "core",
            "input_contract": _INPUT_CONTRACT,
            "grid_contract": "cudnn_batch_norm_backward.kernelgen-dev.core/v1",
            "producer_selector": {"OPS_BASELINE_CUDNN_BN_CURRENT_CORE": "1"},
            "input_builder_sha256": hashlib.sha256(
                inspect.getsource(_build_inputs_fn).encode()
            ).hexdigest(),
        }

    @cached_property
    def _nvidia_baseline(self):
        # Explicit local input; listing/preflight/profile never load this file.
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite baseline JSON constant: {value}")

        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        if (
            data[self.op_name]["native_api"] != f"ops.{self.op_name}"
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["run"]["repo"] != "pytorch"
            or not environment["hardware"]["gpu"]["name"].startswith("NVIDIA H800")
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or not environment["software"]["packages"]["torch"]
            or environment["source"]["git_dirty"] is not False
        ):
            raise ValueError("External baseline needs a completed, traceable H800 run")
        return data[self.op_name]["shapes"], {
            "kind": "external_nvidia",
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "environment": environment,
            "ratio": "raw NVIDIA latency / target latency; no peak normalization",
            "target_timing_mode": base.Config.mode.value,
        }

    def _import_nvidia_latency(self, args, kwargs):
        if kwargs or self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "External baseline supports these forward positional cases only"
            )
        config = self._external_input_config(args)
        shapes, provenance = self._nvidia_baseline
        matches = [
            record
            for dtypes in shapes.values()
            for dtype, record in dtypes.items()
            if dtype == str(args[0].dtype)
            and isinstance(record.get("config"), dict)
            and {
                k: v
                for k, v in record["config"].items()
                if k != "flag_gems_source_commit"
            }
            == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one NVIDIA record matching the actual inputs"
            )
        record = matches[0]
        origin = record["config"].get("flag_gems_source_commit")
        if (
            not isinstance(origin, str)
            or len(origin) != 40
            or any(c not in "0123456789abcdef" for c in origin)
        ):
            raise ValueError("NVIDIA record needs a pinned FlagGems recipe origin")
        latency_us = record.get("T_us")
        if (
            isinstance(latency_us, bool)
            or not isinstance(latency_us, (int, float))
            or not math.isfinite(latency_us)
            or latency_us <= 0
        ):
            raise ValueError("NVIDIA T_us must be finite and positive")
        latency_ms = latency_us / 1000
        recorded_ms = record.get("latency_ms", latency_ms)
        if (
            isinstance(recorded_ms, bool)
            or not isinstance(recorded_ms, (int, float))
            or not math.isfinite(recorded_ms)
            or not math.isclose(recorded_ms, latency_ms, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("NVIDIA latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": record["config"],
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(args[0].device),
                "device_name": base.torch_device_fn.get_device_name(),
                "torch": str(torch.__version__),
            },
        }
        if base.Config.reference_only:
            # The base runner creates all records before traversing inputs.
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError("Cannot identify the active external-reference case")
            active[0]["baseline_source"] = self._imported_baseline
        return latency_ms

    def get_latency(self, op, *args, **kwargs):
        importing = os.environ.get("FLAGGEMS_NV_BASELINE") and (
            base.Config.reference_only
            or getattr(self, "_native_measurement_pending", False)
        )
        if importing:
            if base.Config.skip_native or op is not self.torch_op:
                raise ValueError(
                    "External baseline requires the original reference measurement"
                )
            # Only the first, reference measurement imports. Even an identical
            # candidate callable must execute through the real timer next.
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        self._native_measurement_pending = False
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external:
            if self.is_backward or "tflops" in self.to_bench_metrics:
                raise ValueError(
                    "External baseline supports forward latency measurements only"
                )
            if base.Config.skip_native:
                raise ValueError("Do not combine an external baseline with skip_native")
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if external and metric.latency is not None:
                if (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Target latency must be finite and positive")
            if self._imported_baseline is not None:
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _CrossMachineMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def _run_reference_cases(self, case_ids):
        first_record = len(base.Config.reference_records)
        if os.environ.get("FLAGGEMS_NV_BASELINE") and base.Config.skip_native:
            raise ValueError("Do not combine an external baseline with skip_native")
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            for record in base.Config.reference_records[first_record:]:
                if record.get("baseline_source") and record.get("status") == "PASSED":
                    record["stage"] = "external_baseline"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    if dtype == torch.bfloat16 and (
        flag_gems.vendor_name == "nvidia" or os.environ.get("FLAGGEMS_NV_BASELINE")
    ):
        pytest.skip("native cuDNN bfloat16 backward has no supported NVIDIA baseline")
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.ones(channels, dtype=torch.float32, device=device)
    save_mean = torch.randn(channels, dtype=torch.float32, device=device)
    save_var = torch.rand(channels, dtype=torch.float32, device=device) + 0.1
    reserve = torch.empty((0,), dtype=torch.uint8, device=device)
    return inp, grad, weight, None, None, save_mean, save_var, 1e-5, reserve


@pytest.mark.cudnn_batch_norm_backward
def test_cudnn_batch_norm_backward():
    OperatorCaseBenchmark(
        op_name="cudnn_batch_norm_backward",
        torch_op=torch.ops.aten.cudnn_batch_norm_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
