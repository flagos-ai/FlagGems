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

LDL_FACTOR_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]


@dataclass
class _LdlFactorMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class LdlFactorBenchmark(OperatorBenchmark):
    @cached_property
    def _h800_records(self):
        path = Path(os.environ["FLAGGEMS_LDL_FACTOR_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = path.read_bytes()
        expected_sha = (
            "9a11c5e57869eefdcf01f43889bf340041edd40db78df5dda462565f299792d8"
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError("Expected the reviewed current-core native H800 LDL data")
        data = json.loads(payload)
        key = "linalg_ldl_factor_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[key]["native_api"] != "ops." + key
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "e7178927627a123ec4fcec24cc1cd8d9102b869d"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data["_collection"]["configuration"]["ncu_enabled"] is not False
            or sum(len(ds) for ds in data[key]["shapes"].values()) != 10
        ):
            raise ValueError(
                "Expected the complete ten-record native H800 LDL collection"
            )
        return data[key]["shapes"], dict(
            artifact=str(path),
            artifact_sha256=expected_sha,
            operator=key,
            collection_task=task,
            environment=environment,
            input_identity="Exact current configuration/seed recipe; not cross-device random-byte identity",
            ratio="Raw H800 native latency / locally timed target candidate; no scaling",
            timing_qualification="Shared functional collection, latency only; no NCU metrics",
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_ldl_native_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._ldl_native_pending = False
        if op is not self.torch_op:
            raise ValueError("External LDL latency replaces only its native reference")
        if self.is_backward or len(args) != 1 or kwargs != {"hermitian": False}:
            raise ValueError(
                "Static LDL requires the original forward input and explicit hermitian=False"
            )
        (inp,) = args
        if (
            tuple(inp.shape) not in LDL_FACTOR_SHAPES
            or inp.dtype not in (torch.float32, torch.float64)
            or not inp.is_contiguous()
            or inp.storage_offset() != 0
            or inp.requires_grad
            or inp.device.type != torch.device(self.device).type
        ):
            raise ValueError(
                "Runtime LDL input differs from the collected current-core contract"
            )
        config = dict(
            inputs=dict(
                args=[
                    dict(
                        shape=list(inp.shape),
                        dtype=str(inp.dtype),
                        stride=list(inp.stride()),
                        requires_grad=False,
                    )
                ],
                kwargs=kwargs,
            ),
            source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            benchmark="benchmark/test_linalg_ldl_factor.py",
            scope="core",
            descriptor=list(inp.shape),
            variant=0,
            input_recipe=dict(seed=0, input="torch.randn", diagonal_shift="n*eye(n)"),
        )
        records, source = self._h800_records
        matches = [
            r
            for ds in records.values()
            for dtype, r in ds.items()
            if dtype == str(inp.dtype) and r["config"] == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one native H800 LDL record matching the full runtime configuration"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
            or not math.isclose(
                record["latency_ms"], us / 1000, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "Native LDL latency must be positive with consistent microsecond/millisecond units"
            )
        self._imported_baseline = dict(
            **source, config=config, target_timing_mode=base.Config.mode.value
        )
        return us / 1000

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        self._ldl_native_pending = (
            flag_gems.vendor_name in ("mthreads", "ascend") and needs_native
        )
        self._imported_baseline = None
        try:
            if self._ldl_native_pending and (
                self.is_backward or "tflops" in self.to_bench_metrics
            ):
                raise ValueError(
                    "Static H800 LDL supports the original forward latency measurement"
                )
            if self._ldl_native_pending and not os.environ.get(
                "FLAGGEMS_LDL_FACTOR_H800_BASELINE"
            ):
                raise ValueError("Missing native H800 LDL reference")
            metric = super()._measure_input(input, case_id)
            if self._imported_baseline is not None:
                if metric.latency is not None and (
                    not math.isfinite(metric.latency) or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target LDL candidate latency must be positive and finite"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _LdlFactorMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._ldl_native_pending = False

    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_FACTOR_SHAPES
        self.shape_desc = "N, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"hermitian": False},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    if base.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin cannot preserve float64 input storage")
    if base.vendor_name == "iluvatar" and dtype == torch.float64:
        pytest.skip("Iluvatar native ldl_factor does not support float64")
    torch.manual_seed(0)
    n = shape[-1]
    factory_device = (
        "cpu" if base.vendor_name == "ascend" and dtype == torch.float64 else device
    )
    inp = torch.randn(shape, dtype=dtype, device=factory_device)
    inp = inp @ inp.transpose(-2, -1) + n * torch.eye(
        n, dtype=dtype, device=factory_device
    )
    return inp.to(device), {"hermitian": False}


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
def test_linalg_ldl_factor():
    LdlFactorBenchmark(
        op_name="ldl_factor",
        torch_op=torch.linalg.ldl_factor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float64],
    ).run()
