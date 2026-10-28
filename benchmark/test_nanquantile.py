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

_DEFAULT_SHAPES = [
    (1,),
    (17,),
    (1024,),
    (1025,),
    (128, 128),
    (512, 512),
    (1024, 1024),
]


_H800_SHA256 = "0bd21fe9919f49bbded0429ebea33239520a5b07edf483d7c09b5132d1949397"
_H800_COLLECTION = "f6e5662ff5efddfaa1234170041821116f8c9783"
_H800_ARTIFACT_COMMIT = "293e47b919067b10c25d4443289bbc7828f4ee58"
_RECIPE_SOURCE = "ac9a1f703757537a3d54cc1ac1febeca410316c0"


@dataclass
class _NanquantileMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _H800Latency:
    @cached_property
    def _h800(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _H800_SHA256:
            raise ValueError("Expected the reviewed native H800 nanquantile snapshot")
        data = json.loads(payload)
        if data[self.op_name]["native_api"] != "ops." + self.op_name:
            raise ValueError("Expected the corresponding native nanquantile overload")
        return data, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=_H800_SHA256,
            artifact_git_commit=_H800_ARTIFACT_COMMIT,
            collection_source_commit=_H800_COLLECTION,
            ratio="H800 native latency_ms / locally timed target candidate latency_ms",
            input_identity="Same builders, distributions, NaN locations and constant q; backend RNG payloads are not identical.",
        )

    @staticmethod
    def _describe(value):
        if isinstance(value, torch.Tensor):
            return dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                stride=list(value.stride()),
                requires_grad=value.requires_grad,
            )
        if isinstance(value, (tuple, list)):
            return [_H800Latency._describe(x) for x in value]
        if isinstance(value, dict):
            return {k: _H800Latency._describe(x) for k, x in value.items()}
        return value

    def _input_config(self, args, kwargs):
        inp = args[0]
        scalar_out = self.op_name == "nanquantile_scalar_out"
        return dict(
            inputs=dict(args=self._describe(args), kwargs=self._describe(kwargs)),
            source_commit=_RECIPE_SOURCE,
            benchmark="benchmark/test_nanquantile.py",
            descriptor=list(inp.shape),
            scope="core",
            variant=0,
            input_recipe=dict(
                seed=0,
                input="torch.randn",
                nan_start=0,
                nan_stride=5,
                q=0.73 if scalar_out else [0.1, 0.5, 0.9],
                dim=-1,
                keepdim=False,
                interpolation="linear",
            ),
        )

    def _import_latency(self, args, kwargs):
        scalar_out = self.op_name == "nanquantile_scalar_out"
        if len(args) != (4 if scalar_out else 3) or set(kwargs) != (
            {"out", "interpolation"} if scalar_out else set()
        ):
            raise ValueError("Expected the original nanquantile positional/out ABI")
        inp = args[0]
        if (
            inp.dtype != torch.float32
            or not inp.is_contiguous()
            or inp.requires_grad
            or tuple(inp.shape) not in _DEFAULT_SHAPES
        ):
            raise ValueError(
                "External snapshot requires original dense Float32 core inputs"
            )
        if scalar_out:
            out = kwargs["out"]
            if (
                out.device != inp.device
                or out.dtype != inp.dtype
                or out.shape != inp.shape[:-1]
                or not out.is_contiguous()
                or out.requires_grad
                or out.data_ptr() == inp.data_ptr()
            ):
                raise ValueError(
                    "External snapshot requires the separate original out buffer"
                )
        else:
            q = args[1]
            if (
                q.device != inp.device
                or q.dtype != inp.dtype
                or list(q.shape) != [3]
                or not q.is_contiguous()
                or q.requires_grad
                or not torch.equal(
                    q.cpu(), torch.tensor([0.1, 0.5, 0.9], dtype=q.dtype)
                )
            ):
                raise ValueError(
                    "Tensor q differs from the original constant Float32 recipe"
                )
        # Observe input values on CPU to avoid the configured native Long-sort gap.
        flat = inp.detach().cpu().reshape(-1)
        mask = torch.zeros(flat.numel(), dtype=torch.bool)
        mask[::5] = True
        if (
            not torch.equal(torch.isnan(flat), mask)
            or not torch.isfinite(flat[~mask]).all()
        ):
            raise ValueError(
                "NaN positions differ from the original every-fifth-element recipe"
            )
        data, source = self._h800
        config = self._input_config(args, kwargs)
        encoded = json.dumps(config, sort_keys=True, allow_nan=False)
        rows = [
            row
            for group in data[self.op_name]["shapes"].values()
            for dtype, row in group.items()
            if dtype == str(inp.dtype)
            and json.dumps(row.get("config"), sort_keys=True, allow_nan=False)
            == encoded
        ]
        if len(rows) != 1:
            raise ValueError(
                "Expected one H800 record matching the complete typed input config"
            )
        row = rows[0]
        origin = row["collection_origin"]
        environment, task = origin["environment"], origin["task"]
        gpu, run = environment["hardware"]["gpu"], environment["run"]
        if (
            gpu["name"] != "NVIDIA H800"
            or gpu["uuid"].removeprefix("GPU-")
            != "f71a6457-0dba-3807-5973-e98752701911"
            or environment["source"]["git_commit"] != _H800_COLLECTION
            or environment["source"]["git_dirty"] is not False
            or not environment["software"]["torch_cuda_version"]
            or run["repo"] != "pytorch"
            or run["ncu_enabled"] is not False
            or run["selected_ops"] != [self.op_name]
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or task["device"] != "2"
            or row["metric_validity"] != {"T_us": "measured", "latency_ms": "measured"}
        ):
            raise ValueError(
                "Static latency lacks the exact clean native H800 worker provenance"
            )
        for field in ("T_us", "latency_ms"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("H800 latency must be finite positive measured time")
        milliseconds = row["T_us"] / 1000
        if not math.isclose(
            milliseconds, row["latency_ms"], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("H800 microsecond/millisecond units disagree")
        self._imported_baseline = dict(
            **source,
            config=config,
            collection_origin=origin,
            ncu_enabled=False,
            latency_validity="measured",
            target_timing_mode=base.Config.mode.value,
        )
        return milliseconds

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
            raise ValueError(
                "Enflame nanquantile requires FLAGGEMS_NV_BASELINE native H800 data"
            )
        if external and (
            self.is_backward
            or "tflops" in self.to_bench_metrics
            or base.Config.skip_native
            or base.Config.reference_only
        ):
            raise ValueError(
                "Static nanquantile data requires ordinary forward latency timing"
            )
        self._imported_baseline = None
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if self._imported_baseline is not None:
                value = metric.latency
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "Actual target candidate latency must be finite and positive"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / value
                return _NanquantileMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False


class NanquantileBenchmark(_H800Latency, OperatorBenchmark):
    def set_shapes(self, shape_file_path):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [(33, 65), (4096,), (256, 1024), (4096, 1024)]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "q": [3]},
        params={"dim": -1, "interpolation": "linear"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp.reshape(-1)[::5] = float("nan")
    q = torch.tensor([0.1, 0.5, 0.9], dtype=dtype, device=device)
    return inp, q, -1


@pytest.mark.nanquantile
def test_nanquantile():
    NanquantileBenchmark(
        op_name="nanquantile",
        torch_op=torch.ops.aten.nanquantile.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()


def _scalar_out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "out": list(dims[:-1])},
        params={"q": 0.73, "dim": -1, "keepdim": False, "interpolation": "linear"},
        builder_args=(dims,),
    )


def _scalar_out_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp.reshape(-1)[::5] = float("nan")
    out = torch.empty(shape[:-1], dtype=dtype, device=device)
    return inp, 0.73, -1, False, {"interpolation": "linear", "out": out}


@pytest.mark.nanquantile_scalar_out
def test_nanquantile_scalar_out():
    NanquantileBenchmark(
        op_name="nanquantile_scalar_out",
        torch_op=torch.ops.aten.nanquantile.scalar_out,
        case_fn=_scalar_out_case_fn,
        build_inputs_fn=_scalar_out_build_inputs_fn,
        dtypes=[torch.float32],
        is_inplace=True,
        fresh_inputs=True,
    ).run()
