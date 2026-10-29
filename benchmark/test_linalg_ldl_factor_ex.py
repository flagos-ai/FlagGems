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

LDL_FACTOR_EX_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]


@dataclass
class _LdlMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class LinalgLdlFactorExBenchmark(base.GenericBenchmark):
    @cached_property
    def _h800_baseline(self):
        path = Path(os.environ["FLAGGEMS_LDL_H800_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != (
            "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3"
        ):
            raise ValueError("Expected the reviewed ops-baseline H800 snapshot")
        data = json.loads(payload)
        key = "linalg_ldl_factor_ex"  # Historical ops-baseline API alias.
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            data[key]["native_api"] != "ops." + key
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "ccc0508d81b3b10283a62dab73fc05e256093848"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
        ):
            raise ValueError("Expected the completed native H800 LDL collection")
        return data[key]["shapes"], {
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "operator": key,
            "collection_task": task,
            "hardware": environment["hardware"],
            "software": environment["software"],
            "source": environment["source"],
            "environment_qualification": "Per-device run reflects the last task and is not an LDL-specific receipt",
            "input_identity": "Exact recipe and input contract; random payloads differ across devices",
            "ratio": "raw H800 latency / locally timed target candidate; no peak normalization",
        }

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name != "thead" or op is not self.torch_op:
            return super().get_latency(op, *args, **kwargs)
        if self.is_backward or kwargs or len(args) != 1 or base.Config.skip_native:
            raise ValueError("H800 LDL baseline requires the original forward core ABI")
        (inp,) = args
        if (
            tuple(inp.shape) not in LDL_FACTOR_EX_SHAPES
            or inp.dtype != torch.float32
            or not inp.is_contiguous()
            or inp.requires_grad
            or inp.storage_offset() != 0
            or inp.device.type != torch.device(self.device).type
        ):
            raise ValueError("LDL inputs differ from the collected H800 recipe")
        config = {
            "inputs": {
                "args": [
                    {
                        "shape": list(inp.shape),
                        "dtype": str(inp.dtype),
                        "stride": list(inp.stride()),
                        "requires_grad": False,
                    }
                ],
                "kwargs": {},
            },
            "source_commit": "9d57a23b4386bd669152a935fe88a96db907da7c",
            "benchmark": "benchmark/test_linalg_ldl_factor_ex.py",
            "descriptor": list(inp.shape),
            "scope": "core",
            "variant": 0,
        }
        records, source = self._h800_baseline
        matches = [
            r
            for dtypes in records.values()
            for dtype, r in dtypes.items()
            if dtype == str(inp.dtype) and r["config"] == config
        ]
        if len(matches) != 1:
            raise ValueError("Expected one H800 LDL record for the full input identity")
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 LDL latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 LDL latency units disagree")
        self._imported_baseline = {
            **source,
            "config": config,
            "target_timing_mode": base.Config.mode.value,
        }
        return ms

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "thead" and op is self.torch_op:
            raise ValueError(
                "PPU LDL needs an explicit candidate because native SYTRF is unavailable"
            )
        return op, dispatch, overridden

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name == "thead" and (
            self.is_backward
            or base.Config.skip_native
            or "tflops" in self.to_bench_metrics
        ):
            raise ValueError("H800 LDL comparison requires forward latency measurement")
        self._imported_baseline = None
        metric = super()._measure_input(input, case_id)
        if self._imported_baseline is not None:
            if metric.latency is not None and (
                not math.isfinite(metric.latency) or metric.latency <= 0
            ):
                raise ValueError("Target LDL latency must be finite and positive")
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
            return _LdlMetrics(
                **asdict(metric), baseline_source=self._imported_baseline
            )
        return metric

    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_FACTOR_EX_SHAPES
        self.shape_desc = "N, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    n = shape[-1]
    a = torch.randn(shape, dtype=dtype, device=device)
    return (a @ a.mT + torch.eye(n, dtype=dtype, device=device) * 0.1,)


@pytest.mark.linalg_ldl_factor_ex
@pytest.mark.ldl_factor_ex
def test_linalg_ldl_factor_ex():
    LinalgLdlFactorExBenchmark(
        op_name="ldl_factor_ex",
        torch_op=torch.linalg.ldl_factor_ex,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
