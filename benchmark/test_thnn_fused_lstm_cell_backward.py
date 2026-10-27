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
from .conftest import Config

CORE_SHAPES = [
    (1, 32, False),
    (8, 64, True),
    (32, 128, False),
    (64, 256, True),
    (128, 512, False),
    (256, 1024, True),
]
COMPREHENSIVE_SHAPES = [
    (512, 1024, False),
    (512, 2048, True),
    (1024, 2048, True),
]
DTYPES = list(consts.FLOAT_DTYPES)
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)


def lstm_forward(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    dtype = cx.dtype
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    x = input_gates.detach().cpu().to(compute) + hidden_gates.detach().cpu().to(compute)
    if input_bias is not None:
        x = (
            x
            + input_bias.detach().cpu().to(compute)
            + hidden_bias.detach().cpu().to(compute)
        )
    i, f, g, o = x.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c = f * cx.detach().cpu().to(compute) + i * g
    h = o * c.tanh()
    return h.to(dtype), c.to(dtype), torch.cat((i, f, g, o), 1).to(dtype)


def _forward_workspace(*args):
    # PyTorch CUDA RNN.cu stores activated gates in the original scalar dtype.
    # Enflame has no fused-cell CPU kernel behind its native fallback. Preserve
    # target RNG construction, then prepare the dense packet on CPU explicitly.
    if flag_gems.vendor_name != "enflame":
        return torch.ops.aten._thnn_fused_lstm_cell(*args)
    values = lstm_forward(*args)
    return tuple(value.to(args[0].device) for value in values)


_H800_SHA256 = "60283f9aece55ce65e95c2a7e74a04a46e7f02b1d29ab0c110ba0181a1100c5e"
_COLLECTOR_COMMIT = "e1d48e31e472f23d1d42a6ec2398ba0c1d56d0be"


@dataclass
class _ExternalFusedCellMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _FusedCellStaticBaseline:
    """Import only the native H800 reference timer; target candidate still runs."""

    @cached_property
    def _external_records(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _H800_SHA256:
            raise ValueError("Expected exact reviewed fused-cell H800 artifact")
        data = json.loads(payload)
        baseline_key = self.op_name.lstrip("_") + "_current_core"
        task = data["_collection"]["tasks"][baseline_key]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["hardware"]["gpu"]["uuid"]
            != "fd660a7d-352c-6699-9e93-74c8ea136095"
            or environment["source"]["git_commit"] != _COLLECTOR_COMMIT
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["ncu_enabled"] is not False
            or data[baseline_key]["native_api"] != "ops." + baseline_key
        ):
            raise ValueError("Incomplete native H800 fused-cell provenance")
        expected_count = 3 if self.op_name.endswith("_impl") else 24
        records = data[baseline_key]["shapes"]
        if sum(len(ds) for ds in records.values()) != expected_count:
            raise ValueError("Wrong complete native fused-cell core grid")
        return records, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=_H800_SHA256,
            collection_task=task,
            device_environment=environment,
            ratio="H800 native latency_ms / target candidate latency_ms",
        )

    def _import_latency(self, args, kwargs):
        gru = self.op_name == "_thnn_fused_gru_cell_backward"
        impl = self.op_name == "_thnn_fused_lstm_cell_backward_impl"
        if kwargs or len(args) != (3 if gru else 6):
            raise ValueError("Expected original positional fused-cell backward ABI")
        tensors, bias = args[:-1], args[-1]
        if type(bias) is not bool or not all(
            isinstance(x, torch.Tensor) for x in tensors
        ):
            raise ValueError("Core fused-cell tensors/has_bias differ from recipe")
        grad = tensors[0]
        if grad.device.type != torch.device(self.device).type:
            raise ValueError("Actual fused-cell inputs are not on the benchmark target")
        if grad.ndim != 2:
            raise ValueError("Expected rank-two core state")
        batch, hidden = grad.shape
        expected_shapes = (
            [(batch, hidden), (batch, 5 * hidden)]
            if gru
            else [(batch, hidden)] * 4 + [(batch, 4 * hidden)]
        )
        if any(
            tuple(x.shape) != shape
            or x.dtype != grad.dtype
            or x.device != grad.device
            or not x.is_contiguous()
            or x.requires_grad
            for x, shape in zip(tensors, expected_shapes)
        ):
            raise ValueError(
                "Fused-cell input shape/dtype/device/stride contract mismatch"
            )
        if impl:
            shapes = [(1, 4, True), (4, 16, True), (16, 64, True)]
        else:
            shapes = [
                (1, 32, False),
                (8, 64, True),
                (32, 128, False),
                (64, 256, True),
                (128, 512, False),
                (256, 1024, True),
            ]
        if (batch, hidden, bias) not in shapes:
            raise ValueError("Input does not identify an original core case")
        describe = lambda x: dict(
            shape=list(x.shape),
            dtype=str(x.dtype),
            stride=list(x.stride()),
            requires_grad=x.requires_grad,
        )
        rnn = "gru" if gru else "lstm"
        recipe = {
            "rng": "manual_seed(0)" if impl else "ambient target RNG",
            "input_gates": "randn",
            "hidden_gates": "randn_like(input_gates)",
            "state": "randn",
            "input_bias": (
                "zeros"
                if impl
                else (
                    "randn independent" if gru else "shared randn if has_bias else None"
                )
            ),
            "hidden_bias": (
                "randn"
                if impl
                else "randn_like(input_bias)" if gru else "same object as input_bias"
            ),
            "gradients": (
                "randn(batch,hidden) after forward"
                if gru
                else "randn_like forward outputs"
            ),
        }
        config = dict(
            inputs=dict(args=[describe(x) for x in tensors] + [bias], kwargs={}),
            source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
            benchmark="benchmark/test_thnn_fused_"
            + rnn
            + "_cell_backward"
            + ("_impl" if impl else "")
            + ".py",
            descriptor=[batch, hidden, bias],
            scope="core",
            variant=0,
            native_api="torch.ops.aten." + self.op_name,
            input_recipe=recipe,
            workspace_producer="torch.ops.aten._thnn_fused_"
            + rnn
            + "_cell native CUDA outside timer",
            autograd=False,
        )
        records, source = self._external_records
        matches = [
            r
            for ds in records.values()
            for dtype, r in ds.items()
            if dtype == str(grad.dtype) and r.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError("Expected unique native H800 full input/recipe match")
        record = matches[0]
        latency = record["T_us"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError("Native H800 latency must be finite positive microseconds")
        milliseconds = latency / 1000
        if (
            isinstance(record["latency_ms"], bool)
            or not isinstance(record["latency_ms"], (int, float))
            or not math.isfinite(record["latency_ms"])
            or record["latency_ms"] <= 0
        ):
            raise ValueError("Native latency_ms must be finite positive numeric")
        if not math.isclose(
            record["latency_ms"], milliseconds, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("Inconsistent native H800 latency units")
        self._baseline_source = {
            **source,
            "config": config,
            "target_workspace_producer": (
                "CPU mathematical producer validated against native CUDA, restored to real target"
                if flag_gems.vendor_name == "enflame"
                else "native target forward"
            ),
            "target_environment": dict(
                vendor=flag_gems.vendor_name,
                device=str(grad.device),
                torch=str(torch.__version__),
            ),
        }
        if base.Config.reference_only:
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError("Cannot identify active external fused-cell case")
            active[0]["baseline_source"] = self._baseline_source
        return milliseconds

    def get_latency(self, op, *args, **kwargs):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if getattr(self, "_import_reference_pending", False) or (
            external and base.Config.reference_only
        ):
            self._import_reference_pending = False
            if op is not self.torch_op:
                raise ValueError(
                    "Static native data may replace only the reference timer"
                )
            return self._import_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if flag_gems.vendor_name == "enflame" and not external:
            raise ValueError(
                "Enflame fused-cell timing requires static native H800 data"
            )
        if external and (self.is_backward or base.Config.skip_native):
            raise ValueError(
                "External fused-cell data requires native/candidate latency mode"
            )
        self._baseline_source = None
        self._import_reference_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if self._baseline_source is not None:
                return _ExternalFusedCellMetrics(
                    **asdict(metric), baseline_source=self._baseline_source
                )
            return metric
        finally:
            self._import_reference_pending = False


def _thnn_fused_lstm_cell_backward_no_grad(*args):
    with torch.no_grad():
        return torch.ops.aten._thnn_fused_lstm_cell_backward(*args)


class FusedLSTMCellBackwardBenchmark(_FusedCellStaticBaseline, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_case_iter(self, cur_dtype):
        for ordinal, (batch_size, hidden_size, has_bias) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"grad_hy": [batch_size, hidden_size]},
                    params={"has_bias": has_bias},
                    builder_args=(batch_size, hidden_size, has_bias),
                ),
            )

    def build_inputs(self, case):
        batch_size, hidden_size, has_bias = case.builder_args[0].builder_args
        cur_dtype = case.dtype
        if flag_gems.vendor_name == "ascend" and cur_dtype == torch.bfloat16:
            pytest.skip(
                "Ascend native LSTM forward used to build backward inputs "
                "does not support bfloat16 (input_gates must be float or half)."
            )
        input_gates = torch.randn(
            batch_size,
            4 * hidden_size,
            dtype=cur_dtype,
            device=self.device,
        )
        hidden_gates = torch.randn_like(input_gates)
        cx = torch.randn(batch_size, hidden_size, dtype=cur_dtype, device=self.device)
        bias = (
            torch.randn(4 * hidden_size, dtype=cur_dtype, device=self.device)
            if has_bias
            else None
        )
        with torch.no_grad():
            hy, cy, workspace = _forward_workspace(
                input_gates, hidden_gates, cx, bias, bias
            )
        return torch.randn_like(hy), torch.randn_like(cy), cx, cy, workspace, has_bias


@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.underscore_thnn_fused_lstm_cell_backward
@torch.no_grad()
def test_thnn_fused_lstm_cell_backward():
    bench = FusedLSTMCellBackwardBenchmark(
        op_name="_thnn_fused_lstm_cell_backward",
        torch_op=_thnn_fused_lstm_cell_backward_no_grad,
        dtypes=DTYPES,
    )
    bench.run()
