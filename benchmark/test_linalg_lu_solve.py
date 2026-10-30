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

SHAPES = [(4, 4), (16, 1), (33, 16), (64, 64), (128, 128)]
DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


@dataclass
class _LuSolveMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class LinalgLuSolveBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = SHAPES

    @cached_property
    def _h800_records(self):
        path = Path(os.environ["FLAGGEMS_LU_SOLVE_H800_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        expected_sha = (
            "c24d0eb2f9e62d7e4dcbd4548228807188469b98d68b88a50dea777d71c7742e"
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(
                "Expected the reviewed current-core native H800 LU solve data"
            )
        data = json.loads(payload)
        key = "linalg_lu_solve_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[key]["native_api"] != "ops." + key
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "7ee5adc05a27f39397ead75878f88512ec775c9a"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data["_collection"]["configuration"]["ncu_enabled"] is not False
            or sum(len(ds) for ds in data[key]["shapes"].values()) != 10
        ):
            raise ValueError(
                "Expected the complete ten-record native H800 LU solve collection"
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
        if not getattr(self, "_lu_native_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._lu_native_pending = False
        if op is not self.torch_op or self.is_backward or len(args) != 3 or kwargs:
            raise ValueError(
                "Static LU solve requires the original forward native call"
            )
        lu, pivots, rhs = args
        if lu.ndim != 2 or rhs.ndim != 2:
            raise ValueError("Static LU solve requires unbatched core inputs")
        n, k = rhs.shape
        if (
            (n, k) not in SHAPES
            or lu.shape != (n, n)
            or pivots.shape != (n,)
            or lu.dtype not in (torch.float32, torch.float64)
            or rhs.dtype != lu.dtype
            or pivots.dtype != torch.int32
            or not (lu.device == pivots.device == rhs.device)
            or lu.device.type != torch.device(self.device).type
            or any(t.storage_offset() != 0 or t.requires_grad for t in args)
            or lu.stride() != (1, n)
            or pivots.stride() != (1,)
            or rhs.stride() != (k, 1)
        ):
            raise ValueError(
                "Runtime LU solve inputs differ from the H800 core contract"
            )
        # The original builder omits kwargs; the native schema defaults are
        # left=True and adjoint=False, explicitly recorded by the collector.
        config = dict(
            inputs=dict(
                args=[
                    dict(
                        shape=list(t.shape),
                        dtype=str(t.dtype),
                        stride=list(t.stride()),
                        requires_grad=False,
                    )
                    for t in args
                ],
                kwargs=dict(left=True, adjoint=False),
            ),
            source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            benchmark="benchmark/test_linalg_lu_solve.py",
            scope="core",
            descriptor=[n, k],
            variant=0,
            input_recipe=dict(
                seed=0,
                matrix="torch.randn(n,n); matrix@matrix.mT+n*eye(n)",
                factorization="torch.linalg.lu_factor(matrix)",
                rhs="torch.randn(n,k) after factorization",
            ),
        )
        records, source = self._h800_records
        matches = [
            r
            for ds in records.values()
            for dtype, r in ds.items()
            if dtype == str(lu.dtype) and r["config"] == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one H800 LU solve record matching the full input configuration"
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
                "Native LU solve latency must be positive with consistent units"
            )
        self._imported_baseline = dict(
            **source,
            config=config,
            target_timing_mode=base.Config.mode.value,
            original_runtime_kwargs={},
            resolved_native_defaults=dict(left=True, adjoint=False)
        )
        return us / 1000

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        self._lu_native_pending = flag_gems.vendor_name == "ascend" and needs_native
        self._imported_baseline = None
        try:
            if self._lu_native_pending and (
                self.is_backward or "tflops" in self.to_bench_metrics
            ):
                raise ValueError(
                    "Static H800 LU solve supports the original forward latency measurement"
                )
            if self._lu_native_pending and not os.environ.get(
                "FLAGGEMS_LU_SOLVE_H800_BASELINE"
            ):
                raise ValueError("Missing native H800 LU solve reference")
            metric = super()._measure_input(input, case_id)
            if self._imported_baseline is not None:
                if metric.latency is not None and (
                    not math.isfinite(metric.latency) or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target LU solve candidate latency must be positive and finite"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _LuSolveMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._lu_native_pending = False

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SHAPES)


def _case_fn(shape, dtype):
    del dtype
    n, k = shape
    yield base.BenchmarkCasePlan(
        shape={"LU": [n, n], "pivots": [n], "B": [n, k]},
        params={"left": True, "adjoint": False},
        builder_args=(n, k),
    )


def _build_inputs_fn(plan, dtype, device):
    n, k = plan.builder_args
    torch.manual_seed(0)
    factory_device = "cpu" if flag_gems.vendor_name == "thead" else device
    matrix = torch.randn(n, n, dtype=dtype, device=factory_device)
    matrix = matrix @ matrix.mT + torch.eye(n, dtype=dtype, device=factory_device) * n
    lu, pivots = torch.linalg.lu_factor(matrix)
    rhs = torch.randn(n, k, dtype=dtype, device=factory_device)
    return tuple(value.to(device) for value in (lu, pivots, rhs))


@pytest.mark.linalg_lu_solve
def test_linalg_lu_solve():
    LinalgLuSolveBenchmark(
        op_name="linalg_lu_solve",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.linalg_lu_solve,
        dtypes=DTYPES,
    ).run()
