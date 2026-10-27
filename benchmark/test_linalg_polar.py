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
from functools import cached_property
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

CORE_SHAPES = [(4, 2), (8, 4), (16, 8), (16, 8, 4)]
COMPREHENSIVE_SHAPES = [(64, 8, 4), (32, 16, 8)]


def _native_polar(inp):
    # This is the native PyTorch GPU SVD composition measured by ops-baseline,
    # not the unrelated elementwise torch.polar API or a fused polar primitive.
    u, values, vh = torch.linalg.svd(inp, full_matrices=False)
    polar_u = u @ vh
    polar_h = vh.mH @ (values.unsqueeze(-1) * vh)
    return polar_u, 0.5 * (polar_h + polar_h.mH)


@dataclass
class _PolarMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _PolarBaseline:
    def _external_input_config(self, args, kwargs):
        if len(args) != 1 or set(kwargs) != (
            {"U", "H"} if self.op_name == "linalg_polar_out" else set()
        ):
            raise ValueError("Polar baseline requires the original input/out ABI")
        inp = args[0]
        tensors = (inp, *kwargs.values())
        if any(
            not isinstance(t, torch.Tensor)
            or t.layout != torch.strided
            or t.device != inp.device
            for t in tensors
        ):
            raise ValueError("Polar baseline needs dense tensors on one device")

        def tensor(t):
            return dict(
                shape=list(t.shape),
                dtype=str(t.dtype),
                stride=list(t.stride()),
                requires_grad=t.requires_grad,
            )

        return {
            "inputs": {
                "args": [tensor(inp)],
                "kwargs": {name: tensor(t) for name, t in kwargs.items()},
            },
            "reference_implementation": "pytorch_native_svd_polar_composition",
            "full_matrices": False,
            "side": "right",
            "symmetrize_H": True,
            "input_recipe": "torch.manual_seed(0);torch.randn(shape,dtype=torch.float32,device=device)",
            "out_aliases": ["U", "H"] if kwargs else [],
        }

    @cached_property
    def _nvidia_baseline(self):
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

        def load(payload):
            return json.loads(
                payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
            )

        data = load(payload)
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        artifact_hash = hashlib.sha256(payload).hexdigest()
        proof = load(path.with_suffix(".provenance.json").read_bytes())
        if (
            data[self.op_name]["native_api"] != f"ops.{self.op_name}"
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["run"]["repo"] != "pytorch"
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or environment["source"]["git_dirty"] is not False
            or not environment["software"]["packages"]["torch"]
            or proof["status"] != "PASS_VERIFIED_NATIVE_H800_SNAPSHOT"
            or proof["artifact_sha256"] != artifact_hash
            or proof["collection_source_commit"] != commit
            or proof["source_clean"] is not True
            or proof["controller_environment"]["NVIDIA_TF32_OVERRIDE"] != "0"
        ):
            raise ValueError("Polar baseline needs a verified full-precision H800 run")
        return data[self.op_name]["shapes"], {
            "kind": "external_nvidia",
            "artifact": str(path),
            "artifact_sha256": artifact_hash,
            "environment": environment,
            "reference_implementation": "pytorch_native_svd_polar_composition",
            "input_identity": "same distribution/configuration, not identical random payloads",
            "ratio": "raw H800 SVD-composition latency / target latency",
            "target_timing_mode": base.Config.mode.value,
        }

    def _import_nvidia_latency(self, args, kwargs):
        config = self._external_input_config(args, kwargs)
        shapes, provenance = self._nvidia_baseline
        matches = [
            record
            for by_dtype in shapes.values()
            for dtype, record in by_dtype.items()
            if dtype == str(args[0].dtype) and record.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError("Expected exactly one matching H800 polar configuration")
        row = matches[0]
        value = row["T_us"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("H800 latency must be finite and positive")
        latency = value / 1000
        if not math.isclose(row["latency_ms"], latency, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(args[0].device),
                "torch": str(torch.__version__),
            },
        }
        return latency

    def get_latency(self, op, *args, **kwargs):
        if getattr(self, "_native_measurement_pending", False):
            if op is not self.torch_op:
                raise ValueError("External baseline must replace the reference timer")
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        # Even an identical candidate callable goes through the real local timer.
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external and (
            self.is_backward
            or "tflops" in self.to_bench_metrics
            or base.Config.skip_native
            or base.Config.reference_only
        ):
            raise ValueError(
                "Polar external baseline requires ordinary forward latency timing"
            )
        self._imported_baseline = None
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if self._imported_baseline is not None:
                if (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target candidate latency must be finite and positive"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _PolarMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False


class LinalgPolarBenchmark(_PolarBaseline, OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_SHAPES)
        self.shape_desc = "batch dimensions, rows, columns"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"A": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    return (torch.randn(shape, dtype=dtype, device=device),)


@pytest.mark.linalg_polar
def test_linalg_polar():
    LinalgPolarBenchmark(
        op_name="linalg_polar",
        torch_op=_native_polar,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
