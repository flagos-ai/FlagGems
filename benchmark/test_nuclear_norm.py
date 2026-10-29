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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]


def _input_config(shape):
    return {
        "inputs": {
            "args": [
                {
                    "shape": list(shape),
                    "dtype": "torch.float32",
                    "stride": [shape[-1], 1],
                    "requires_grad": False,
                }
            ],
            "kwargs": {"keepdim": False},
        },
        "grid_contract": "flaggems-ac9-core6",
        "reference_implementation": "torch.nuclear_norm",
        "input_recipe": "torch.manual_seed(0);torch.randn(shape,dtype=torch.float32,device=device)",
        "output_aliases": [],
    }


@dataclass
class _NuclearNormMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class OperatorCaseBenchmark(OperatorBenchmark):
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

        def load(content):
            return json.loads(
                content, object_pairs_hook=unique_keys, parse_constant=reject_constant
            )

        data = load(payload)
        task = data["_collection"]["tasks"]["nuclear_norm"]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        proof = load(path.with_suffix(".provenance.json").read_bytes())
        artifact_hash = hashlib.sha256(payload).hexdigest()
        if (
            data["nuclear_norm"]["native_api"] != "ops.nuclear_norm"
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
            or proof["controller_environment"]["OPS_BASELINE_NUCLEAR_NORM_CURRENT_CORE"]
            != "1"
        ):
            raise ValueError("nuclear_norm requires a completed native H800 snapshot")
        shapes = data["nuclear_norm"]["shapes"]
        if len(shapes) != len(SHAPES) or any(
            set(per_dtype) != {"torch.float32"} for per_dtype in shapes.values()
        ):
            raise ValueError("H800 snapshot must contain only the six float32 cases")
        rows = [per_dtype["torch.float32"] for per_dtype in shapes.values()]
        wanted = [_input_config(shape) for shape in SHAPES]
        if len(rows) != len(wanted) or any(
            sum(row.get("config") == config for row in rows) != 1 for config in wanted
        ):
            raise ValueError("H800 nuclear_norm snapshot must match all six core cases")
        # Reject an incomplete or invalid snapshot before consuming any case.
        for row in rows:
            value = row["T_us"]
            latency = row["latency_ms"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or not math.isclose(latency, value / 1000, rel_tol=1e-9, abs_tol=1e-12)
            ):
                raise ValueError(
                    "Every H800 core latency must be finite with valid units"
                )
        return rows, {
            "kind": "external_nvidia",
            "artifact": str(path),
            "artifact_sha256": artifact_hash,
            "environment": environment,
            "reference_implementation": "torch.nuclear_norm",
            "input_identity": "same seeded distribution/configuration, not identical device RNG payloads",
            "ratio": "raw H800 native nuclear_norm latency / local target candidate latency",
            "target_timing_mode": base.Config.mode.value,
        }

    def _import_nvidia_latency(self, args, kwargs):
        if len(args) != 1 or kwargs != {"keepdim": False}:
            raise ValueError("Expected the original nuclear_norm core ABI")
        inp = args[0]
        if (
            not isinstance(inp, torch.Tensor)
            or inp.dtype != torch.float32
            or tuple(inp.shape) not in SHAPES
            or inp.layout != torch.strided
            or not inp.is_contiguous()
            or inp.requires_grad
        ):
            raise ValueError(
                "Expected an original float32 contiguous nuclear_norm input"
            )
        config = _input_config(inp.shape)
        rows, provenance = self._nvidia_baseline
        matches = [row for row in rows if row.get("config") == config]
        if len(matches) != 1:
            raise ValueError("Expected one matching H800 nuclear_norm record")
        row = matches[0]
        value = row["T_us"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("H800 latency must be positive and finite")
        latency = value / 1000
        if not math.isclose(row["latency_ms"], latency, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(inp.device),
                "torch": str(torch.__version__),
            },
        }
        return latency

    def get_latency(self, op, *args, **kwargs):
        if getattr(self, "_native_measurement_pending", False):
            if op is not self.torch_op:
                raise ValueError(
                    "Static baseline must replace only the reference timer"
                )
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        # An identical bootstrap callable still takes the real local timer.
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if flag_gems.vendor_name == "mthreads" and not external:
            raise ValueError(
                "MUSA nuclear_norm requires FLAGGEMS_NV_BASELINE from H800"
            )
        if external and (
            self.is_backward
            or "tflops" in self.to_bench_metrics
            or base.Config.skip_native
            or base.Config.reference_only
        ):
            raise ValueError(
                "External nuclear_norm baseline requires forward latency timing"
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
                        "Target candidate latency must be positive and finite"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _NuclearNormMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(*B), M, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"keepdim": False},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    return (inp, {"keepdim": False})


@pytest.mark.nuclear_norm
def test_nuclear_norm():
    OperatorCaseBenchmark(
        op_name="nuclear_norm",
        torch_op=torch.nuclear_norm,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
