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

import flag_gems
import pytest
import torch

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(4, 8), (64, 128), (256, 256), (512, 1024), (2048, 1024)]
_PACK_MAGIC = 260821.0
_PACK_VERSION = 1.0


def _torch_reference(weight, weight_scale, weight_zero_point, bias):
    quantized = torch.clamp(
        torch.round(weight / weight_scale) + weight_zero_point, -128, 127
    ).to(torch.int8)
    metadata = torch.stack(
        (
            weight_scale.to(torch.float32),
            weight_zero_point.to(torch.float32),
            weight_scale.new_tensor(_PACK_MAGIC),
            weight_scale.new_tensor(_PACK_VERSION),
        )
    )
    return torch.cat(
        (
            metadata.contiguous().view(torch.uint8),
            bias.to(torch.float32).contiguous().view(torch.uint8),
            quantized.contiguous().view(torch.uint8).flatten(),
        )
    )


_H800_SHA = "b20a29b92caa8d58a74a02c1e1f1877ba08ccd7489101c3954e3afcdbb3a47fd"
_H800_SOURCE = "e56db9741c416f50cf6e5b30f942c5c6eaa22920"
_COLLECTION_OP = "wrapped_linear_prepack_portable_core"


@dataclass
class _ExternalMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class WrappedLinearPrepackBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "OUT_FEATURES, IN_FEATURES"


    @cached_property
    def _external_records(self):
            path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != _H800_SHA:
                raise ValueError("Expected the reviewed current-core H800 snapshot")
            data = json.loads(payload)
            task = data["_collection"]["tasks"][_COLLECTION_OP]
            environment = data["_environments"][str(task["device"])]
            if (task["status"] != "completed" or task["returncode"] != 0
                    or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
                    or environment["source"]["git_commit"] != _H800_SOURCE
                    or environment["source"]["git_dirty"] is not False
                    or environment["run"]["ncu_enabled"] is not False
                    or data[_COLLECTION_OP]["native_api"] != "ops." + _COLLECTION_OP):
                raise ValueError("Incomplete clean H800 native collection provenance")
            records = data[_COLLECTION_OP]["shapes"]
            if sum(len(ds) for ds in records.values()) != 5:
                raise ValueError("Expected all five original portable H800 records")
            return records, dict(kind="external_nvidia", artifact=str(path),
                                 artifact_sha256=_H800_SHA, collection_task=task,
                                 device_environment=environment,
                                 ratio="H800 original portable reference composition latency_ms / target candidate latency_ms")

    def get_latency(self, op, *args, **kwargs):
            external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
            if getattr(self, "_import_reference_pending", False) or (external and base.Config.reference_only):
                self._import_reference_pending = False
                if op is not self.torch_op:
                    raise ValueError("Static H800 data may replace only the reference timer")
                return self._import_latency(args, kwargs)
            return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
            external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
            if flag_gems.vendor_name == "enflame" and not external:
                raise ValueError("Enflame portable packing timing requires static H800 native data")
            if external and (self.is_backward or base.Config.skip_native):
                raise ValueError("External H800 data requires native/candidate latency mode")
            self._baseline_source = None
            self._import_reference_pending = external and "latency_base" in self.to_bench_metrics
            try:
                metric = super()._measure_input(input, case_id)
                if self._baseline_source is not None:
                    return _ExternalMetrics(**asdict(metric), baseline_source=self._baseline_source)
                return metric
            finally:
                self._import_reference_pending = False

    def _import_latency(self, args, kwargs):
        if len(args) != 4 or kwargs:
            raise ValueError("Expected original portable packing four-input ABI")
        weight, scale, zero_point, bias = args
        if (not all(isinstance(x, torch.Tensor) for x in args)
                or weight.ndim != 2 or bias.ndim != 1 or bias.shape[0] != weight.shape[0]
                or weight.dtype != torch.float32 or bias.dtype != torch.float32
                or scale.dtype != torch.float32 or zero_point.dtype != torch.int64
                or scale.ndim != 0 or zero_point.ndim != 0
                or any(x.device != weight.device or not x.is_contiguous() or x.requires_grad for x in args)
                or scale.item() != 0.03125 or zero_point.item() != -3):
            raise ValueError("External baseline requires the original portable input recipe")
        def describe(x):
            return dict(shape=list(x.shape), dtype=str(x.dtype),
                        stride=list(x.stride()), requires_grad=x.requires_grad)
        config = dict(inputs=dict(args=[describe(x) for x in args], kwargs={}),
                      source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
                      benchmark="benchmark/test_wrapped_linear_prepack.py", scope="core",
                      descriptor=list(weight.shape), variant=0,
                      input_recipe=dict(seed=0,input="torch.randn",bias="torch.randn",scale=0.03125,zero_point=-3),
                      reference_kind="portable_uint8_pytorch_cuda_composition")
        records, source = self._external_records
        matches = [r for ds in records.values() for dtype, r in ds.items()
                   if dtype == str(weight.dtype) and r.get("config") == config]
        if len(matches) != 1:
            raise ValueError("Expected one H800 portable record matching complete typed input config")
        record = matches[0]
        latency = record["T_us"]
        if (isinstance(latency, bool) or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency <= 0
                or not math.isclose(record["latency_ms"], latency / 1000,
                                    rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError("Invalid H800 latency or microsecond/millisecond units")
        self._baseline_source = dict(**source, config=config,
                                    reference_kind="original portable packing PyTorch CUDA composition; not opaque ATen",
                                    target_environment=dict(vendor=flag_gems.vendor_name,
                                                            device=str(weight.device),
                                                            torch=str(torch.__version__)))
        return latency / 1000


def _case_fn(shape, dtype):
    del dtype
    out_features, in_features = shape
    yield base.BenchmarkCasePlan(
        shape={"weight": [out_features, in_features], "bias": [out_features]},
        params={"scale": 0.03125, "zero_point": -3},
        builder_args=(out_features, in_features),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    out_features, in_features = plan.builder_args
    torch.manual_seed(0)
    return (
        torch.randn((out_features, in_features), dtype=torch.float32, device=device),
        torch.tensor(0.03125, dtype=torch.float32, device=device),
        torch.tensor(-3, dtype=torch.int64, device=device),
        torch.randn(out_features, dtype=torch.float32, device=device),
    )


@pytest.mark.wrapped_linear_prepack
@pytest.mark.underscore_wrapped_linear_prepack
def test_wrapped_linear_prepack_benchmark():
    WrappedLinearPrepackBenchmark(
        op_name="_wrapped_linear_prepack",
        torch_op=_torch_reference,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
