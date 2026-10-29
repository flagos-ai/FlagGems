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

_MUSA_CORE_SHAPES = {
    "linalg_cond": [(n, n) for n in (4, 8, 16, 32, 64, 128)],
    "linalg_pinv": [(b, n, n) for b in (1, 4, 8, 16, 32) for n in (4, 8)],
    "linalg_svd": [(n, n) for n in (8, 16, 32, 64)],
    "linalg_svdvals": [(n, n) for n in (16, 32, 64, 128, 256, 512)],
}


@dataclass
class _MusaLinalgMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _MusaLinalgStaticBaseline:
    @cached_property
    def _h800_baseline(self):
        historical = self.op_name == "linalg_pinv"
        path = Path(
            os.environ[
                (
                    "FLAGGEMS_LINALG_PINV_H800_BASELINE"
                    if historical
                    else "FLAGGEMS_MUSA_LINALG_H800_BASELINE"
                )
            ]
        ).resolve(strict=True)
        payload = path.read_bytes()
        expected = (
            "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3"
            if historical
            else "8ba73fbe61042343df6d84dcebcd9c087a882ed5b678b7867f531100ce2a5c77"
        )
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Expected the exact qualified native H800 artifact")
        data = json.loads(payload)
        key = self.op_name if historical else self.op_name + "_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        collected = (
            "ccc0508d81b3b10283a62dab73fc05e256093848"
            if historical
            else "6b674c0daf4ea87fd44dd2ed5fb6436880df1bb2"
        )
        if (
            data[key]["native_api"] != "ops." + key
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != collected
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
        ):
            raise ValueError("Native H800 provenance does not match this operator")
        return data[key]["shapes"], {
            "artifact": str(path),
            "artifact_sha256": expected,
            "operator": key,
            "collection_task": task,
            "environment": environment,
            "environment_qualification": (
                "Historical per-device environment is the last task, not an operator-specific receipt"
                if historical
                else "Actual shared H800 collection; latency only, no NCU"
            ),
            "input_identity": "Exact native API/full configuration/recipe; random payloads differ across devices",
            "ratio": "Raw H800 latency / actual local candidate latency; no peak normalization",
        }

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name != "mthreads" or op is not self.torch_op:
            return super().get_latency(op, *args, **kwargs)
        if (
            self.is_backward
            or base.Config.skip_native
            or "tflops" in self.to_bench_metrics
        ):
            raise ValueError(
                "Static linalg reference supports original forward latency only"
            )
        if self.op_name not in _MUSA_CORE_SHAPES:
            raise ValueError("This artifact does not cover the requested operator")
        positional_none = self.op_name == "linalg_cond"
        if len(args) != (2 if positional_none else 1) or (
            positional_none and args[1] is not None
        ):
            raise ValueError(
                "Native linalg positional ABI differs from the collected recipe"
            )
        expected_kwargs = (
            {"full_matrices": False} if self.op_name == "linalg_svd" else {}
        )
        if kwargs != expected_kwargs or (
            self.op_name == "linalg_svd" and type(kwargs["full_matrices"]) is not bool
        ):
            raise ValueError("Native linalg kwargs differ from the collected recipe")
        inp = args[0]
        if (
            not isinstance(inp, torch.Tensor)
            or tuple(inp.shape) not in _MUSA_CORE_SHAPES[self.op_name]
            or inp.dtype != torch.float32
            or not inp.is_contiguous()
            or inp.storage_offset() != 0
            or inp.requires_grad
            or inp.device.type != torch.device(self.device).type
        ):
            raise ValueError(
                "Native linalg tensor contract differs from the collected recipe"
            )
        historical = self.op_name == "linalg_pinv"
        tensors = [
            {
                "shape": list(inp.shape),
                "dtype": str(inp.dtype),
                "stride": list(inp.stride()),
                "requires_grad": False,
            }
        ]
        if positional_none:
            tensors.append(None)
        config = {
            "inputs": {"args": tensors, "kwargs": kwargs},
            "source_commit": (
                "9d57a23b4386bd669152a935fe88a96db907da7c"
                if historical
                else "2a3a455fa9eef530dc11a6b96ff333ffb47235fd"
            ),
            "benchmark": "benchmark/test_" + self.op_name + ".py",
            "descriptor": list(inp.shape),
            "scope": "core",
            "variant": 0,
        }
        if not historical:
            config["input_recipe"] = {"seed": 0, "input": "torch.randn"}
            if positional_none:
                config["input_recipe"]["diagonal_shift"] = "n*eye(n)"
        records, provenance = self._h800_baseline
        matches = [
            r
            for dtypes in records.values()
            for dtype, r in dtypes.items()
            if dtype == str(inp.dtype) and r["config"] == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one native H800 record for the full input identity"
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
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_timing_mode": base.Config.mode.value,
        }
        return ms

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "mthreads" and op is self.torch_op:
            raise ValueError("Missing native MUSA linalg needs an explicit candidate")
        return op, dispatch, overridden

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        metric = super()._measure_input(input, case_id)
        if self._imported_baseline is None:
            return metric
        if metric.latency is not None and (
            not math.isfinite(metric.latency) or metric.latency <= 0
        ):
            raise ValueError(
                "Local linalg candidate latency must be finite and positive"
            )
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        return _MusaLinalgMetrics(
            **asdict(metric), baseline_source=self._imported_baseline
        )


LINALG_COND_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]


class LinalgCondBenchmark(_MusaLinalgStaticBaseline, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_COND_SHAPES
        self.shape_desc = "N, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"p": None},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        n = shape[-1]
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        matrix = matrix + n * torch.eye(n, dtype=case.dtype, device=self.device)
        return matrix, None


@pytest.mark.linalg_cond
def test_linalg_cond():
    bench = LinalgCondBenchmark(
        op_name="linalg_cond",
        torch_op=torch.linalg.cond,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_cond)
    bench.run()
