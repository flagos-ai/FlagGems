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

INVERSE_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)
INVERSE_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 8, 8),
    (512, 16, 16),
    (128, 32, 32),
    (32, 64, 64),
    (8, 128, 128),
]


@dataclass
class _InverseH800Metrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _InverseH800Baseline:
    @cached_property
    def _inverse_h800_records(self):
        path = Path(os.environ["FLAGGEMS_INVERSE_H800_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        digest = "44667f228420a447f0bfd6b5a8789b39a459f9511396af97463b6fa5f0f8f849"
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Expected the reviewed native H800 inverse core artifact")
        data = json.loads(payload)
        key = "inverse_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        records = data[key]["shapes"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[key]["native_api"] != "ops." + key
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "473500bca2d83dadeada3549904cee96f3281642"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data["_collection"]["configuration"]["ncu_enabled"] is not False
            or sum(len(values) for values in records.values()) != 10
        ):
            raise ValueError("Expected all ten traceable native H800 inverse records")
        return records, dict(
            artifact=str(path),
            artifact_sha256=digest,
            operator=key,
            collection_task=task,
            environment=environment,
            input_identity="Same per-input manual_seed(0) factory and complete configuration; not cross-device random-byte identity",
            ratio="Raw native H800 latency / locally timed candidate; no scaling",
            timing_qualification="Shared functional collection, mean latency only; no NCU metrics",
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_inverse_h800_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._inverse_h800_pending = False
        if op is not self.torch_op or self.is_backward or len(args) != 1 or kwargs:
            raise ValueError("Static inverse requires the original forward native call")
        inp = args[0]
        if (
            tuple(inp.shape) not in INVERSE_SHAPES
            or inp.dtype != torch.float32
            or inp.device.type != torch.device(self.device).type
            or inp.storage_offset() != 0
            or inp.requires_grad
            or not inp.is_contiguous()
        ):
            raise ValueError("Runtime inverse input differs from H800 core contract")
        expected = dict(
            inputs=dict(
                args=[
                    dict(
                        shape=list(inp.shape),
                        dtype=str(inp.dtype),
                        stride=list(inp.stride()),
                        requires_grad=False,
                    )
                ],
                kwargs={},
            ),
            source_commit="f29c81779afe5668a3f021aa36bb879021e2b709",
            benchmark="benchmark/test_inverse.py",
            scope="core",
            descriptor=list(inp.shape),
            variant=0,
            input_recipe=dict(
                matrix="eye(n) expanded contiguous for batch +0.05*torch.randn(shape,dtype,device)",
                rng_policy="torch.manual_seed(0) for every input build before randn",
                native="torch.inverse(input)",
            ),
        )
        records, source = self._inverse_h800_records
        matches = [
            record
            for values in records.values()
            for dtype, record in values.items()
            if dtype == str(inp.dtype)
            and {
                key: value
                for key, value in record["config"].items()
                if key != "producer_sample"
            }
            == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one H800 record for the full inverse configuration"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 inverse latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 inverse latency units disagree")
        self._inverse_h800_source = dict(source, config=record["config"])
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if flag_gems.vendor_name != "kunlunxin" or not needs_native:
            return super()._measure_input(input, case_id)
        if not os.environ.get("FLAGGEMS_INVERSE_H800_BASELINE"):
            raise ValueError(
                "Kunlun native inverse has a matrix-size limit; set FLAGGEMS_INVERSE_H800_BASELINE to the reviewed native H800 core artifact"
            )
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Static inverse baseline requires the original forward latency measurement"
            )
        self._inverse_h800_pending = "latency_base" in self.to_bench_metrics
        self._inverse_h800_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._inverse_h800_source is not None:
                if metric.latency is not None and (
                    isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target inverse candidate must have a real positive latency"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _InverseH800Metrics(
                    **asdict(metric), baseline_source=self._inverse_h800_source
                )
            return metric
        finally:
            self._inverse_h800_pending = False


class InverseBenchmark(_InverseH800Baseline, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = INVERSE_SHAPES
        self.shape_desc = "batch..., N, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        n = shape[-1]
        identity = torch.eye(n, dtype=case.dtype, device=self.device)
        if len(shape) > 2:
            identity = identity.expand(shape).contiguous()
        matrix = identity + 0.05 * torch.randn(
            shape, dtype=case.dtype, device=self.device
        )
        return (matrix,)


@pytest.mark.inverse
def test_inverse():
    bench = InverseBenchmark(
        op_name="inverse",
        torch_op=torch.inverse,
        dtypes=INVERSE_DTYPES,
    )
    bench.set_gems(flag_gems.inverse)
    bench.run()
