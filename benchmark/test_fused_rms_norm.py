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


def _fused_rms_norm_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _fused_rms_norm_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    M, N = shape
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    return inp, (N,), weight


def _fused_rms_norm_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _fused_rms_norm_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    M, N = shape
    normalized_shape = [N]
    eps = 1e-5
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn_like(inp)
    weight = torch.randn(N, dtype=dtype, device=device)
    rstd = torch.rsqrt(inp.to(torch.float32).pow(2).mean(dim=-1) + eps)
    return grad, inp, normalized_shape, rstd, weight, [True, True]


def _torch_fused_rms_norm_backward(
    grad_out, inp, normalized_shape, rstd, weight, output_mask
):
    # Resolve only when called: some Torch backends expose no fused primitive.
    native = getattr(torch.ops.aten, "_fused_rms_norm_backward", None)
    if native is not None:
        return native(grad_out, inp, normalized_shape, rstd, weight, output_mask)
    batch_ndim = inp.ndim - len(normalized_shape)
    norm_dims = tuple(range(batch_ndim, inp.ndim))
    value = inp.to(torch.float32)
    inv_rms = rstd.to(torch.float32).reshape(
        inp.shape[:batch_ndim] + (1,) * len(normalized_shape)
    )
    normalized = value * inv_rms
    grad_input = None
    if output_mask[0]:
        grad = grad_out.to(torch.float32)
        if weight is not None:
            grad = grad * weight.to(torch.float32)
        row_sum = (normalized * grad).sum(dim=norm_dims, keepdim=True)
        grad_input = (
            (grad - normalized / math.prod(normalized_shape) * row_sum) * inv_rms
        ).to(inp.dtype)
    grad_weight = None
    if output_mask[1] and weight is not None:
        grad_weight = (
            (grad_out.to(torch.float32) * normalized)
            .sum(dim=tuple(range(batch_ndim)))
            .reshape(normalized_shape)
            .to(inp.dtype)
        )
    return grad_input, grad_weight


@dataclass
class _RmsBackwardMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _RmsBackwardBaseline:
    # This helper belongs to the two RMS backward benchmarks only. The original
    # input builders and RNG streams stay intact; records match the same input
    # distribution/contract, not byte-identical random payloads across devices.
    baseline_benchmark = "benchmark/test_fused_rms_norm.py"

    def _external_input_config(self, args):
        if len(args) != 6:
            raise ValueError("RMS backward baseline needs its six positional inputs")
        grad, inp, normalized_shape, rstd, weight, output_mask = args
        tensors = (grad, inp, rstd, weight)
        if any(
            not isinstance(t, torch.Tensor) or t.layout != torch.strided
            for t in tensors
        ):
            raise ValueError("RMS backward baseline requires dense tensors and weight")
        if any(t.device != inp.device for t in tensors):
            raise ValueError("RMS backward baseline inputs must share one device")

        def tensor(t):
            return {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "stride": list(t.stride()),
                "requires_grad": t.requires_grad,
            }

        return {
            "inputs": {
                "args": [
                    tensor(grad),
                    tensor(inp),
                    list(normalized_shape),
                    tensor(rstd),
                    tensor(weight),
                    list(output_mask),
                ],
                "kwargs": {},
            },
            "source_commit": "d20405a359ab31f2c5d856c760850486d858e341",
            "benchmark": self.baseline_benchmark,
            "descriptor": list(inp.shape),
            "scope": "core",
            "variant": 0,
            "rstd_recipe": "rsqrt(mean(input.float()**2,dim=-1)+1e-5)",
            "reference_implementation": "pytorch_native_operator_composition",
        }

    @cached_property
    def _nvidia_baseline(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite baseline JSON constant: {value}")

        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        key = (
            "fused_rms_norm_backward"  # ops-baseline omits the ATen leading underscore.
        )
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        if (
            data[key]["native_api"] != f"ops.{key}"
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["run"]["repo"] != "pytorch"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or environment["source"]["git_dirty"] is not False
            or not environment["software"]["packages"]["torch"]
        ):
            raise ValueError(
                "RMS backward baseline needs a completed, traceable H800 run"
            )
        return data[key]["shapes"], {
            "kind": "external_nvidia",
            "operator": key,
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "environment": environment,
            "reference_implementation": "pytorch_native_operator_composition",
            "input_identity": "same distribution and input contract; not byte-identical random payloads",
            "ratio": "raw H800 latency / target latency; no peak normalization",
            "target_timing_mode": base.Config.mode.value,
        }

    def _import_nvidia_latency(self, args, kwargs):
        if kwargs or self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "External RMS backward baseline supports the original positional core cases only"
            )
        config = self._external_input_config(args)
        shapes, provenance = self._nvidia_baseline
        matches = [
            record
            for dtypes in shapes.values()
            for dtype, record in dtypes.items()
            if dtype == str(args[1].dtype) and record.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one H800 record matching the actual RMS backward inputs"
            )
        record = matches[0]
        latency_us = record.get("T_us")
        if (
            isinstance(latency_us, bool)
            or not isinstance(latency_us, (int, float))
            or not math.isfinite(latency_us)
            or latency_us <= 0
        ):
            raise ValueError("H800 T_us must be finite and positive")
        latency_ms = latency_us / 1000
        recorded_ms = record.get("latency_ms", latency_ms)
        if (
            isinstance(recorded_ms, bool)
            or not isinstance(recorded_ms, (int, float))
            or not math.isfinite(recorded_ms)
            or not math.isclose(recorded_ms, latency_ms, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(args[1].device),
                "device_name": base.torch_device_fn.get_device_name(),
                "torch": str(torch.__version__),
            },
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
                raise RuntimeError(
                    "Cannot identify the active RMS external-reference case"
                )
            active[0]["baseline_source"] = self._imported_baseline
        return latency_ms

    def get_latency(self, op, *args, **kwargs):
        importing = os.environ.get("FLAGGEMS_NV_BASELINE") and (
            base.Config.reference_only
            or getattr(self, "_native_measurement_pending", False)
        )
        if importing:
            if base.Config.skip_native or op is not self.torch_op:
                raise ValueError(
                    "External baseline requires the original reference measurement"
                )
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        if (
            op is self.torch_op
            and not os.environ.get("FLAGGEMS_NV_BASELINE")
            and getattr(torch.ops.aten, "_fused_rms_norm_backward", None) is None
        ):
            raise ValueError(
                "Missing native RMS backward: set FLAGGEMS_NV_BASELINE to the H800 ops-baseline artifact"
            )
        # Candidate timing always executes locally, even if its callable equals
        # the reference; only the first reference measurement may import.
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external and (
            self.is_backward
            or "tflops" in self.to_bench_metrics
            or base.Config.skip_native
        ):
            raise ValueError(
                "External RMS baseline requires forward latency measurement without skip_native"
            )
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if (
                external
                and metric.latency is not None
                and (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                )
            ):
                raise ValueError("Target latency must be finite and positive")
            if self._imported_baseline is not None:
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _RmsBackwardMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def _run_reference_cases(self, case_ids):
        first_record = len(base.Config.reference_records)
        if os.environ.get("FLAGGEMS_NV_BASELINE") and base.Config.skip_native:
            raise ValueError("Do not combine an external baseline with skip_native")
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            for record in base.Config.reference_records[first_record:]:
                if record.get("baseline_source") and record.get("status") == "PASSED":
                    record["stage"] = "external_baseline"


class _RmsBackwardBenchmark(_RmsBackwardBaseline, base.GenericBenchmark2DOnly):
    pass


@pytest.mark.fused_rms_norm
def test_fused_rms_norm():
    """Benchmark for _fused_rms_norm which returns (output, inv_rms)."""

    def _torch_fused_rms_norm(x, normalized_shape, weight=None, eps=1e-5):
        """Reference implementation for comparison."""
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        hidden_states = upcast_x * inv_rms
        hidden_states = hidden_states.to(x.dtype)
        if weight is not None:
            return weight * hidden_states, inv_rms.squeeze(-1)
        return hidden_states, inv_rms.squeeze(-1)

    bench = base.GenericBenchmark2DOnly(
        case_fn=_fused_rms_norm_case_fn,
        build_inputs_fn=_fused_rms_norm_build_inputs_fn,
        op_name="fused_rms_norm",
        torch_op=_torch_fused_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._fused_rms_norm)
    bench.run()


@pytest.mark.fused_rms_norm_backward
@pytest.mark.underscore_fused_rms_norm_backward
def test_fused_rms_norm_backward():
    """Benchmark for aten._fused_rms_norm_backward which returns (dx, dw)."""

    bench = _RmsBackwardBenchmark(
        case_fn=_fused_rms_norm_backward_case_fn,
        build_inputs_fn=_fused_rms_norm_backward_build_inputs_fn,
        op_name="_fused_rms_norm_backward",
        torch_op=_torch_fused_rms_norm_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
