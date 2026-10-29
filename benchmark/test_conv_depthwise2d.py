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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

CASES = [
    (32, 4, 8, 8, 2, 2, 1, 0, 1),
    (18, 16, 4, 4, 2, 2, 2, 1, 1),
    (1, 64, 224, 224, 3, 3, 1, 1, 1),
    (1, 128, 112, 112, 5, 5, 2, 2, 1),
]

_H800_SHA = "994c96eb30805b319672606cbe6e5bdbeab266a4bce3dd0eeb6b7c75262eeebb"
_H800_SOURCE = "4b01b9b2480f6d0ac99aef8c3d45c79029918b39"
_COLLECTION_OP = "conv_depthwise2d_current_core"


@dataclass
class _ExternalMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class ConvDepthwise2dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "N, C, H, W, Kh, Kw, stride, padding, dilation"

    @cached_property
    def _external_records(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _H800_SHA:
            raise ValueError("Expected the reviewed current-core H800 snapshot")
        data = json.loads(payload)
        task = data["_collection"]["tasks"][_COLLECTION_OP]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _H800_SOURCE
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["ncu_enabled"] is not False
            or data[_COLLECTION_OP]["native_api"] != "ops." + _COLLECTION_OP
        ):
            raise ValueError("Incomplete clean H800 native collection provenance")
        records = data[_COLLECTION_OP]["shapes"]
        if sum(len(ds) for ds in records.values()) != 12:
            raise ValueError("Expected all twelve current-core H800 records")
        return records, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=_H800_SHA,
            collection_task=task,
            device_environment=environment,
            ratio="H800 native latency_ms / target candidate latency_ms",
        )

    def _import_latency(self, args, kwargs):
        if len(args) != 7 or kwargs:
            raise ValueError("Expected original depthwise positional ABI")
        inp, weight, kernel, bias, stride, padding, dilation = args
        if (
            not isinstance(inp, torch.Tensor)
            or not isinstance(weight, torch.Tensor)
            or inp.ndim != 4
            or weight.ndim != 4
            or bias is not None
            or inp.device != weight.device
            or inp.dtype != weight.dtype
            or not inp.is_contiguous()
            or not weight.is_contiguous()
            or inp.requires_grad
            or weight.requires_grad
            or list(weight.shape[:2]) != [inp.shape[1], 1]
            or list(weight.shape[-2:]) != list(kernel)
            or stride[0] != stride[1]
            or padding[0] != padding[1]
            or dilation[0] != dilation[1]
        ):
            raise ValueError(
                "External baseline requires the original dense input recipe"
            )
        descriptor = [*inp.shape, *kernel, stride[0], padding[0], dilation[0]]

        def describe(x):
            return (
                dict(
                    shape=list(x.shape),
                    dtype=str(x.dtype),
                    stride=list(x.stride()),
                    requires_grad=x.requires_grad,
                )
                if isinstance(x, torch.Tensor)
                else x
            )

        config = dict(
            inputs=dict(args=[describe(x) for x in args], kwargs={}),
            source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
            benchmark="benchmark/test_conv_depthwise2d.py",
            scope="core",
            descriptor=descriptor,
            variant=0,
            input_recipe=dict(
                seed=0, input="torch.randn", weight="torch.randn", bias=None
            ),
        )
        records, source = self._external_records
        matches = [
            r
            for ds in records.values()
            for dtype, r in ds.items()
            if dtype == str(inp.dtype) and r.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one H800 record matching complete typed input config"
            )
        record = matches[0]
        latency = record["T_us"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
            or not math.isclose(
                record["latency_ms"], latency / 1000, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError("Invalid H800 latency or microsecond/millisecond units")
        self._baseline_source = dict(
            **source,
            config=config,
            target_environment=dict(
                vendor=flag_gems.vendor_name,
                device=str(inp.device),
                torch=str(torch.__version__),
            ),
        )
        return latency / 1000

    def get_latency(self, op, *args, **kwargs):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if getattr(self, "_import_reference_pending", False) or (
            external and base.Config.reference_only
        ):
            self._import_reference_pending = False
            if op is not self.torch_op:
                raise ValueError(
                    "Static H800 data may replace only the reference timer"
                )
            return self._import_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if flag_gems.vendor_name in ("enflame", "mthreads") and not external:
            raise ValueError(
                "Backend depthwise timing requires static H800 native data"
            )
        if external and (self.is_backward or base.Config.skip_native):
            raise ValueError(
                "External H800 data requires native/candidate latency mode"
            )
        self._baseline_source = None
        self._import_reference_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if self._baseline_source is not None:
                return _ExternalMetrics(
                    **asdict(metric), baseline_source=self._baseline_source
                )
            return metric
        finally:
            self._import_reference_pending = False


def _case_fn(shape, dtype):
    del dtype
    case = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"case": list(case)}, params={}, builder_args=(case,)
    )


def _build_inputs_fn(plan, dtype, device):
    n, c, h, w, kh, kw, stride, padding, dilation = plan.builder_args[0]
    torch.manual_seed(0)
    inp = torch.randn((n, c, h, w), dtype=dtype, device=device)
    weight = torch.randn((c, 1, kh, kw), dtype=dtype, device=device)
    return (
        inp,
        weight,
        [kh, kw],
        None,
        [stride, stride],
        [padding, padding],
        [dilation, dilation],
    )


@pytest.mark.conv_depthwise2d
@pytest.mark.underscore_conv_depthwise2d
def test_conv_depthwise2d():
    ConvDepthwise2dBenchmark(
        op_name="_conv_depthwise2d",
        torch_op=torch.ops.aten._conv_depthwise2d.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
