# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

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

SHAPES = [(4, 32, 256, 256), (4, 64, 512, 512), (8, 64, 256, 256)]


NATIVE_BASELINE_RECIPE = {
    "source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0",
    "source_file": "benchmark/test_quantized_lstm.py",
    "seed": 0,
    "quantized_engine": "fbgemm",
    "weight_distribution": "randn * 0.3",
    "bias_distribution": "randn * 0.1",
    "weight_qscheme": "per_tensor_affine",
    "weight_qdtype": "torch.qint8",
    "weight_zero_point": 0,
    "weight_scale": "max(abs(weight).max().item() / 127, 1e-8)",
    "rounding": "linear_prepack -> linear_unpack -> dequantize -> input dtype",
    "native_expression": "torch._VF.lstm(value, hx, deq, *flags)",
    "packed_params": "CPU ScriptObjects used by candidate only, excluded from original floating native reference"
}


@dataclass
class _ExternalBaselineMetrics(base.consts.BenchmarkMetrics):
    baseline_source: dict = None


class _ExternalLstmH800Baseline:
    @staticmethod
    def _input_description(value):
        if isinstance(value, torch.Tensor):
            result = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "stride": list(value.stride()),
                "requires_grad": value.requires_grad,
            }
            if value.is_quantized:
                result.update(
                    qscheme=str(value.qscheme()),
                    scale=value.q_scale(),
                    zero_point=value.q_zero_point(),
                )
            return result
        if isinstance(value, (tuple, list)):
            return [_ExternalLstmH800Baseline._input_description(x) for x in value]
        if isinstance(value, dict):
            return {
                k: _ExternalLstmH800Baseline._input_description(v) for k, v in value.items()
            }
        return value

    @cached_property
    def _h800_baseline(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline key: {key}")
                result[key] = value
            return result

        def finite(value):
            raise ValueError(f"Non-finite baseline constant: {value}")

        expected_sha = os.environ["FLAGGEMS_NV_BASELINE_SHA256"]
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError("H800 baseline artifact SHA256 differs from its frozen receipt")
        data = json.loads(payload, object_pairs_hook=unique, parse_constant=finite)
        aliases = ("quantized_lstm",)
        baseline_operator = next((name for name in aliases if name in data), None)
        if baseline_operator is None:
            raise ValueError(
                "Expected the original dequantized-weight floating LSTM native baseline"
            )
        task = data["_collection"]["tasks"][baseline_operator]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[baseline_operator]["native_api"] != f"ops.{baseline_operator}"
            or environment["run"]["repo"] != "pytorch"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_dirty"] is not False
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or not environment["software"]["packages"]["torch"]
        ):
            raise ValueError(
                "Baseline must be a completed, traceable native H800 collection"
            )
        return data[baseline_operator]["shapes"], {
            "kind": "external_nvidia",
            "operator": self.op_name,
            "native_operator": baseline_operator,
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "environment": environment,
            "input_identity": "same distribution and full input contract, not byte-identical random payloads",
            "ratio": "raw H800 latency / locally measured target candidate latency",
        }

    def get_latency(self, op, *args, **kwargs):
        if os.environ.get("FLAGGEMS_NV_BASELINE") and getattr(
            self, "_external_native_pending", False
        ):
            if op is not self.torch_op:
                raise ValueError(
                    "External latency may replace only the original native reference"
                )
            self._external_native_pending = False
            if len(args) != 10 or kwargs:
                raise ValueError("Expected the original packed/dequantized LSTM benchmark ABI")
            # The original floating native reference ignores packed params.
            # Keep all arguments it actually consumes, including deq and flags.
            native_args = args[:2] + args[3:]
            expected = {
                "args": self._input_description(native_args),
                "kwargs": self._input_description(kwargs),
            }
            shapes, provenance = self._h800_baseline
            matches = [
                record
                for dtypes in shapes.values()
                for record in dtypes.values()
                if record.get("config", {}).get("inputs") == expected
                and record.get("config", {}).get("case_id") == self._external_case_id
                and record.get("config", {}).get("recipe") == NATIVE_BASELINE_RECIPE
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Expected exactly one H800 record for the full actual input configuration"
                )
            record = matches[0]
            us = record.get("T_us")
            if (
                isinstance(us, bool)
                or not isinstance(us, (int, float))
                or not math.isfinite(us)
                or us <= 0
            ):
                raise ValueError("H800 latency must be finite and positive")
            ms = us / 1000
            recorded_ms = record.get("latency_ms", ms)
            if (
                isinstance(recorded_ms, bool)
                or not isinstance(recorded_ms, (int, float))
                or not math.isfinite(recorded_ms)
                or not math.isclose(recorded_ms, ms, rel_tol=1e-9, abs_tol=1e-12)
            ):
                raise ValueError("H800 latency units disagree")
            self._external_baseline_source = {**provenance, "config": record["config"]}
            return ms
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._external_case_id = case_id
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external and (
            base.Config.skip_native
            or self.is_backward
            or "tflops" in self.to_bench_metrics
        ):
            raise ValueError(
                "External H800 baseline requires the original forward latency measurement"
            )
        self._external_native_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        self._external_baseline_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._external_baseline_source is not None:
                if (
                    metric.latency is None
                    or isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target candidate must still have a real finite positive latency"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _ExternalBaselineMetrics(
                    **asdict(metric), baseline_source=self._external_baseline_source
                )
            return metric
        finally:
            self._external_native_pending = False



class QuantizedLstmBenchmark(_ExternalLstmH800Baseline, OperatorBenchmark):
    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if overridden:
            # deq belongs only to the native floating-point reference. The
            # injected quantized_lstm keeps its packed-parameter public ABI.
            def candidate(value, hx, params, deq, *args, **kwargs):
                return op(value, hx, params, *args, **kwargs)

            return candidate, dispatch, overridden
        return op, dispatch, overridden

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "seq,batch,input,hidden"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    seq, batch, inp, hidden = dims
    yield base.BenchmarkCasePlan(
        shape={"input": [seq, batch, inp], "hx": [1, batch, hidden]},
        params={"layers": 1, "bidirectional": False}, builder_args=(dims,),
    )


def _pack(input_size, hidden_size, device, dtype):
    original_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = "fbgemm"
    try:
        w_ih = torch.randn(4 * hidden_size, input_size) * 0.3
        w_hh = torch.randn(4 * hidden_size, hidden_size) * 0.3
        b_ih = torch.randn(4 * hidden_size) * 0.1
        b_hh = torch.randn(4 * hidden_size) * 0.1
        q_ih = torch.quantize_per_tensor(w_ih, max(w_ih.abs().max().item() / 127, 1e-8), 0, torch.qint8)
        q_hh = torch.quantize_per_tensor(w_hh, max(w_hh.abs().max().item() / 127, 1e-8), 0, torch.qint8)
        packed_ih = torch.ops.quantized.linear_prepack(q_ih, b_ih)
        packed_hh = torch.ops.quantized.linear_prepack(q_hh, b_hh)
        params = torch.ops.quantized.make_quantized_cell_params_dynamic(
            packed_ih, packed_hh, b_ih, b_hh, True
        )
        deq = [torch.ops.quantized.linear_unpack(packed_ih)[0].dequantize().to(device=device, dtype=dtype),
               torch.ops.quantized.linear_unpack(packed_hh)[0].dequantize().to(device=device, dtype=dtype),
               b_ih.to(device=device, dtype=dtype), b_hh.to(device=device, dtype=dtype)]
        return params, deq
    finally:
        torch.backends.quantized.engine = original_engine


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    seq, batch, inp_size, hidden = plan.builder_args[0]
    value = torch.randn(seq, batch, inp_size, dtype=dtype, device=device)
    hx = [torch.randn(1, batch, hidden, dtype=dtype, device=device),
          torch.randn(1, batch, hidden, dtype=dtype, device=device)]
    params, deq = _pack(inp_size, hidden, device, dtype)
    return value, hx, [params], deq, True, 1, 0.0, False, False, False


def _torch_lstm(value, hx, params, deq, *args):
    del params
    return torch._VF.lstm(value, hx, deq, *args)


@pytest.mark.quantized_lstm
def test_quantized_lstm():
    bench = QuantizedLstmBenchmark(
        op_name="quantized_lstm", torch_op=_torch_lstm,
        case_fn=_case_fn, build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(lambda value, hx, params, deq, *args: flag_gems.quantized_lstm(value, hx, params, *args))
    bench.run()
