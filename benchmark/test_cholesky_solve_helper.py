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

CHOLESKY_SOLVE_SHAPES = [(4, 1), (8, 2), (16, 4), (32, 4), (64, 8)]


@dataclass
class _CholeskyH800Metrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _CholeskyH800Baseline:
    @cached_property
    def _cholesky_h800_records(self):
        path = Path(os.environ["FLAGGEMS_CHOLESKY_SOLVE_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = path.read_bytes()
        digest = "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3"
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(
                "Expected the published native H800 Cholesky solve artifact"
            )
        data = json.loads(payload)
        task = data["_collection"]["tasks"]["cholesky_solve_helper"]
        environment = data["_environments"][str(task["device"])]
        operator = data["cholesky_solve_helper"]
        records = operator["shapes"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or operator["native_api"] != "ops.cholesky_solve_helper"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "ccc0508d81b3b10283a62dab73fc05e256093848"
            or environment["source"]["git_dirty"] is not False
            or sum(len(values) for values in records.values()) != 5
        ):
            raise ValueError(
                "Expected five traceable native H800 Cholesky solve records"
            )
        return records, dict(
            artifact=str(path),
            artifact_sha256=digest,
            operator="cholesky_solve_helper",
            collection_task=task,
            environment=environment,
            input_identity="Same matrix construction and full input contract; not cross-device random-byte identity",
            ratio="Raw native H800 latency / locally timed candidate; no scaling",
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_cholesky_h800_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._cholesky_h800_pending = False
        if op is not self.torch_op or self.is_backward or len(args) != 3 or kwargs:
            raise ValueError("Static Cholesky solve requires the original forward call")
        rhs, factor, upper = args
        if (
            type(upper) is not bool
            or upper
            or rhs.ndim != 2
            or (rhs.shape[0], rhs.shape[1]) not in CHOLESKY_SOLVE_SHAPES
            or factor.shape != (rhs.shape[0], rhs.shape[0])
            or rhs.dtype != torch.float32
            or factor.dtype != rhs.dtype
            or factor.device != rhs.device
            or rhs.device.type != torch.device(self.device).type
            or any(
                value.storage_offset() != 0 or value.requires_grad for value in args[:2]
            )
        ):
            raise ValueError(
                "Runtime inputs differ from the H800 Cholesky solve contract"
            )
        expected = dict(
            inputs=dict(
                args=[
                    dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        stride=list(value.stride()),
                        requires_grad=False,
                    )
                    for value in args[:2]
                ]
                + [False],
                kwargs={},
            ),
            source_commit="9d57a23b4386bd669152a935fe88a96db907da7c",
            benchmark="benchmark/test_cholesky_solve_helper.py",
            descriptor=list(rhs.shape),
            scope="core",
            variant=0,
        )
        records, source = self._cholesky_h800_records
        matches = [
            record
            for values in records.values()
            for dtype, record in values.items()
            if dtype == str(rhs.dtype) and record["config"] == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one full-configuration H800 Cholesky solve record"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._cholesky_h800_source = dict(source, config=record["config"])
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if flag_gems.vendor_name != "ascend" or not needs_native:
            return super()._measure_input(input, case_id)
        if not os.environ.get("FLAGGEMS_CHOLESKY_SOLVE_H800_BASELINE"):
            raise ValueError(
                "Ascend Cholesky solve uses CPU fallback; set "
                "FLAGGEMS_CHOLESKY_SOLVE_H800_BASELINE to the published H800 artifact"
            )
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Static baseline requires the original forward latency measurement"
            )
        self._cholesky_h800_pending = True
        self._cholesky_h800_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._cholesky_h800_source is not None:
                if metric.latency is not None and (
                    isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Candidate must have a real positive latency")
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _CholeskyH800Metrics(
                    **asdict(metric), baseline_source=self._cholesky_h800_source
                )
            return metric
        finally:
            self._cholesky_h800_pending = False


class CholeskySolveHelperBenchmark(_CholeskyH800Baseline, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_SHAPES
        self.shape_desc = "N, NRHS"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            n, nrhs = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"rhs": [n, nrhs], "factor": [n, n]},
                    params={"upper": False},
                    builder_args=(n, nrhs),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        n, nrhs = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn((n, n), dtype=case.dtype, device=self.device)
        matrix = matrix @ matrix.mT + n * torch.eye(
            n, dtype=case.dtype, device=self.device
        )
        factor = torch.linalg.cholesky(matrix)
        rhs = torch.randn((n, nrhs), dtype=case.dtype, device=self.device)
        return rhs, factor, False


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
def test_cholesky_solve_helper():
    bench = CholeskySolveHelperBenchmark(
        op_name="_cholesky_solve_helper",
        torch_op=torch.ops.aten._cholesky_solve_helper,
        dtypes=[torch.float32],
    )
    bench.run()
