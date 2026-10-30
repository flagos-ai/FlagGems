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

CORE_SHAPES = [
    (2, 2),
    (3, 2),
    (2, 3),
    (4, 4),
    (8, 4),
    (4, 8),
    (256, 2),
    (2, 256),
    (32, 4, 4),
]
COMPREHENSIVE_SHAPES = [
    (1024, 2),
    (2, 1024),
    (256, 4),
    (4, 256),
    (128, 4, 4),
]


@dataclass
class _PinverseMetrics(consts.BenchmarkMetrics):
    baseline_source: dict = None


class _PinverseH800Reference:
    @cached_property
    def _h800_reference(self):
        path = Path(os.environ["FLAGGEMS_PINVERSE_H800_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "8ba73fbe61042343df6d84dcebcd9c087a882ed5b678b7867f531100ce2a5c77":
            raise ValueError("Expected the qualified native H800 pinverse artifact")
        data = json.loads(payload)
        name = "pinverse_current_core"
        task = data["_collection"]["tasks"][name]
        environment = data["_environments"][str(task["device"])]
        if (
            data[name]["native_api"] != "ops." + name
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "6b674c0daf4ea87fd44dd2ed5fb6436880df1bb2"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
        ):
            raise ValueError(
                "Expected a completed, clean native H800 pinverse collection"
            )
        return data[name]["shapes"], {
            "artifact": str(path),
            "artifact_sha256": digest,
            "operator": name,
            "collection_task": task,
            "environment": environment,
            "environment_qualification": (
                "Shared collection per-device envelope; "
                "not an operator-specific startup receipt"
            ),
            "input_identity": "Same full input contract and seed recipe; not identical cross-device payloads",
            "ratio": "Raw H800 latency / actual local candidate latency",
        }

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_pinverse_native_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._pinverse_native_pending = False
        if op is not self.torch_op:
            raise ValueError(
                "External pinverse latency replaces only its native reference"
            )
        if len(args) != 1 or kwargs != {"rcond": 1e-15}:
            raise ValueError(
                "Pinverse positional and rcond ABI must match the native record"
            )
        inp = args[0]
        if (
            not isinstance(inp, torch.Tensor)
            or tuple(inp.shape) not in CORE_SHAPES
            or inp.dtype != torch.float32
            or not inp.is_contiguous()
            or inp.storage_offset() != 0
            or inp.requires_grad
            or inp.device.type != torch.device(self.device).type
        ):
            raise ValueError(
                "Pinverse tensor contract differs from the native H800 record"
            )
        expected = {
            "inputs": {
                "args": [
                    {
                        "shape": list(inp.shape),
                        "dtype": str(inp.dtype),
                        "stride": list(inp.stride()),
                        "requires_grad": False,
                    }
                ],
                "kwargs": {"rcond": 1e-15},
            },
            "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            "benchmark": "benchmark/test_pinverse.py",
            "scope": "core",
            "descriptor": list(inp.shape),
            "variant": 0,
            "input_recipe": {"seed": 0, "input": "torch.randn"},
        }
        shapes, provenance = self._h800_reference
        rows = [
            r
            for dtypes in shapes.values()
            for dtype, r in dtypes.items()
            if dtype == str(inp.dtype) and r["config"] == expected
        ]
        if len(rows) != 1:
            raise ValueError(
                "Expected one native H800 record for the complete pinverse input"
            )
        record = rows[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 pinverse latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 pinverse latency units disagree")
        self._pinverse_imported_reference = {**provenance, "config": expected}
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        self._pinverse_native_pending = (
            flag_gems.vendor_name == "mthreads" and needs_native
        )
        self._pinverse_imported_reference = None
        if self._pinverse_native_pending and (
            self.is_backward or "tflops" in self.to_bench_metrics
        ):
            raise ValueError(
                "The native H800 pinverse artifact covers forward latency only"
            )
        if self._pinverse_native_pending and not os.environ.get(
            "FLAGGEMS_PINVERSE_H800_BASELINE"
        ):
            raise ValueError("Missing native H800 pinverse reference")
        try:
            metric = super()._measure_input(input, case_id)
            if self._pinverse_imported_reference is None:
                return metric
            return _PinverseMetrics(
                **asdict(metric), baseline_source=self._pinverse_imported_reference
            )
        finally:
            self._pinverse_native_pending = False


class PinverseBenchmark(_PinverseH800Reference, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()
        self.shape_desc = "batch..., M, N"

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"rcond": 1e-15},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        return matrix, {"rcond": 1e-15}


@pytest.mark.pinverse
def test_pinverse():
    bench = PinverseBenchmark(
        op_name="pinverse",
        torch_op=torch.pinverse,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.pinverse)
    bench.run()
