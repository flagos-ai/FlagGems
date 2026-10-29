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

_TEST_DTYPES = (
    [torch.float32, torch.float64]
    if flag_gems.vendor_name == "nvidia"
    else [torch.float32]
)


_H800_ARTIFACT_SHA256 = (
    "83290005fa78033935a0fb806843739a41e533741c4611c1239796ae2467f0f3"
)
_H800_OPS_COMMIT = "9b60ea86a30e3ca27fe47abd7efd655499afe791"
_H800_BAD_SHAPES = {(256, 256), (512, 512), (1024, 1024)}


@dataclass
class _LuMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _LuH800PartialBaseline:

    @cached_property
    def _h800_baseline(self):
        path = Path(os.environ["FLAGGEMS_LU_WITH_INFO_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _H800_ARTIFACT_SHA256:
            raise ValueError(
                "Expected the reviewed deprecated LU three-output H800 artifact"
            )
        data = json.loads(payload)
        key = "_lu_with_info"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            data[key]["native_api"] != "ops._lu_with_info"
            or task["status"] != "completed"
            or task["returncode"] != 0
            or (environment["hardware"]["gpu"]["name"] != "NVIDIA H800")
            or (environment["source"]["git_commit"] != _H800_OPS_COMMIT)
            or (environment["source"]["git_dirty"] is not False)
            or (data["_collection"]["configuration"]["repo"] != "pytorch")
        ):
            raise ValueError(
                "Expected a completed native deprecated LU three-output H800 run"
            )
        return (
            data[key]["shapes"],
            {
                "artifact": str(path),
                "artifact_sha256": _H800_ARTIFACT_SHA256,
                "operator": key,
                "collection_task": task,
                "environment": environment,
                "input_identity": "Exact original geometry, flags, dtype and recipe; random payloads differ",
                "ratio": "raw H800 latency / locally timed candidate; no peak normalization",
                "environment_qualification": (
                    "Functional latency collection with sharing allowed; "
                    "no isolated performance claim"
                ),
            },
        )

    def get_latency(self, op, *args, **kwargs):
        if (
            flag_gems.vendor_name != "thead"
            or op is not self.torch_op
            or not args
            or args[0].dtype != torch.float32
            or tuple(args[0].shape) not in _H800_BAD_SHAPES
        ):
            return super().get_latency(op, *args, **kwargs)
        if args[0].device.type != torch.device(self.device).type:
            raise ValueError("H800 comparison requires actual target-device inputs")
        if (
            self.is_backward
            or base.Config.skip_native
            or "tflops" in self.to_bench_metrics
        ):
            raise ValueError(
                "LU-with-info H800 baseline requires original forward core timing"
            )
        config = _h800_input_config(args, kwargs)
        shapes, provenance = self._h800_baseline
        matches = [
            record
            for by_dtype in shapes.values()
            for (dtype, record) in by_dtype.items()
            if dtype == str(args[0].dtype) and record.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one three-output H800 record for the full core identity"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (float, int))
            or (not math.isfinite(us))
            or (us <= 0)
        ):
            raise ValueError("H800 latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-09, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_timing_mode": base.Config.mode.value,
        }
        return ms

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        metric = super()._measure_input(input, case_id)
        if self._imported_baseline is None:
            return metric
        if metric.latency is not None and (
            not math.isfinite(metric.latency) or metric.latency <= 0
        ):
            raise ValueError("Target candidate latency must be finite and positive")
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        return _LuMetrics(**asdict(metric), baseline_source=self._imported_baseline)


def _h800_input_config(args, kwargs):
    if (
        len(args) != 1
        or set(kwargs) != {"pivot", "check_errors"}
        or kwargs["pivot"] is not True
        or kwargs["check_errors"] is not False
    ):
        raise ValueError("Expected original LU core pivot/check_errors flags")
    inp = args[0]
    shape, dtype = tuple(inp.shape), inp.dtype
    if (
        shape not in _H800_BAD_SHAPES
        or dtype != torch.float32
        or not inp.is_contiguous()
        or inp.storage_offset() != 0
        or inp.requires_grad
    ):
        raise ValueError("Expected original failing PPU LU core input identity")
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return {
        "native_expression": "torch._lu_with_info",
        "native_schema": (
            "aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) "
            "-> (Tensor LU, Tensor pivots, Tensor info)"
        ),
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "source_benchmark_sha256": "9d3c5f655d248422d67371653b93d0d8d64c08f3570ca91cce29f190f8f7be49",
        "source_builder_ast_sha256": "62e04bcce6bc7da3aa44e0fe9f7664402d68457fff63fca0bc2490f0dedae2d5",
        "benchmark": "benchmark/test_lu_with_info.py",
        "builder": "LuWithInfoBenchmark.build_inputs",
        "scope": "core",
        "descriptor": list(shape),
        "recipe": {"seed": 0, "input": "torch.randn on requested device"},
        "inputs": {
            "args": [
                {
                    "shape": list(shape),
                    "stride": stride,
                    "storage_offset": 0,
                    "dtype": str(dtype),
                    "requires_grad": False,
                }
            ],
            "kwargs": {"pivot": True, "check_errors": False},
        },
        "outputs": [
            {"name": "LU", "shape": list(shape), "dtype": str(dtype)},
            {
                "name": "pivots",
                "shape": list(shape[:-2]) + [min(shape[-2:])],
                "dtype": "torch.int32",
            },
            {"name": "info", "shape": list(shape[:-2]), "dtype": "torch.int32"},
        ],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }


class LuWithInfoBenchmark(_LuH800PartialBaseline, base.Benchmark):
    DEFAULT_SHAPES = [
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (8, 128, 128),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = "input shape"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"self": list(shape)},
                    params={"pivot": True, "check_errors": False},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, {"pivot": True, "check_errors": False}


@pytest.mark.lu_with_info
@pytest.mark.underscore_lu_with_info
def test_lu_with_info():
    LuWithInfoBenchmark(
        op_name="_lu_with_info",
        torch_op=torch._lu_with_info,
        dtypes=_TEST_DTYPES,
    ).run()
