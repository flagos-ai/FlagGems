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
from functools import cached_property
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark
from .test_linalg_cond import (
    _MusaLinalgStaticBaseline,
    _aten_qualify_metadata,
    _aten_read_bytes,
)

SHAPES = [(8, 8), (16, 16), (32, 32), (64, 64)]


class OperatorCaseBenchmark(_MusaLinalgStaticBaseline, OperatorBenchmark):
    @cached_property
    def _h800_baseline(self):
        path = Path(os.environ["FLAGGEMS_LINALG_EIG_H800_BASELINE"]).resolve(strict=True)
        source_id = "legacy:pytorch-h800-primary"
        payload = _aten_read_bytes(path, source_id)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3":
            raise ValueError("Expected the exact qualified native H800 artifact")
        data = json.loads(payload)
        task = data["_collection"]["tasks"][self.op_name]
        env = data["_environments"][str(task["device"])]
        bank = data[self.op_name]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != "ccc0508d81b3b10283a62dab73fc05e256093848"
            or env["source"]["git_dirty"] is not False
            or bank["native_api"] != "ops.linalg_eig"
            or data["_collection"]["configuration"]["repo"] != "pytorch"
        ):
            raise ValueError("Native H800 provenance does not match linalg_eig")
        return bank["shapes"], _aten_qualify_metadata(
            {
                "artifact": str(path),
                "artifact_sha256": digest,
                "operator": self.op_name,
                "collection_task": task,
                "environment": env,
                "ratio": "Raw H800 latency / actual local candidate latency",
            },
            source_id,
        )

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name != "mthreads" or op is not self.torch_op:
            return OperatorBenchmark.get_latency(self, op, *args, **kwargs)
        if self.is_backward or base.Config.skip_native or "tflops" in self.to_bench_metrics:
            raise ValueError("Static eig reference supports original forward latency only")
        if len(args) != 1 or kwargs:
            raise ValueError("Native eig ABI differs from the collected recipe")
        (inp,) = args
        if (
            not isinstance(inp, torch.Tensor)
            or tuple(inp.shape) not in SHAPES
            or inp.dtype != torch.float32
            or not inp.is_contiguous()
            or inp.storage_offset() != 0
            or inp.requires_grad
            or inp.device.type != torch.device(self.device).type
        ):
            raise ValueError("Native eig tensor differs from the collected recipe")
        config = {
            "inputs": {
                "args": [{
                    "shape": list(inp.shape),
                    "dtype": str(inp.dtype),
                    "stride": list(inp.stride()),
                    "requires_grad": False,
                }],
                "kwargs": {},
            },
            "source_commit": "9d57a23b4386bd669152a935fe88a96db907da7c",
            "benchmark": "benchmark/test_linalg_eig.py",
            "descriptor": list(inp.shape),
            "scope": "core",
            "variant": 0,
        }
        records, provenance = self._h800_baseline
        record = records[str(list(inp.shape))][str(inp.dtype)]
        if record["config"] != config:
            raise ValueError("H800 eig record does not match the full input configuration")
        latency = record["T_us"] / 1000
        if not math.isfinite(latency) or latency <= 0 or not math.isclose(
            record["latency_ms"], latency, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("H800 eig latency must be finite, positive and in milliseconds")
        self._imported_baseline = {
            **provenance, "config": config, "target_timing_mode": base.Config.mode.value
        }
        return latency

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(*B), M, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    return (inp,)


@pytest.mark.linalg_eig
@pytest.mark.skip_native(
    vendors=("thead",), reason="PPU native eig terminates during factorization"
)
def test_linalg_eig():
    OperatorCaseBenchmark(
        op_name="linalg_eig",
        torch_op=torch.linalg.eig,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
