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

import math
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch

from . import base
from .generated_operator_utils import OperatorBenchmark

CASES = [
    ("dense", 1, 2, 2, 128, 128, 64, False, 0.0, False, None, None),
    ("dense", 2, 4, 4, 256, 256, 64, True, 0.0, False, None, None),
    ("dense", 1, 8, 2, 128, 512, 128, False, 0.0, False, None, None),
    ("dense", 1, 8, 1, 1, 1024, 128, False, 0.0, False, None, None),
    ("dense", 1, 4, 4, 128, 512, 128, False, 0.0, False, 128, 0),
    ("dense", 1, 2, 2, 128, 128, 64, False, 0.2, True, None, None),
    ("varlen", 2, 2, 8, 9, 5, 5, 64, False, 0.0, False, None, None),
]


@dataclass
class _ExternalBaselineMetrics(base.consts.BenchmarkMetrics):
    baseline_source: dict = None


class _ExternalH800Baseline:
    @staticmethod
    def _input_description(value):
        if isinstance(value, torch.Tensor):
            result = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "stride": list(value.stride()),
                "requires_grad": value.requires_grad,
            }
            if value.dtype == torch.int32 and value.ndim == 1:
                result["values"] = value.cpu().tolist()
            if value.is_quantized:
                result.update(
                    qscheme=str(value.qscheme()),
                    scale=value.q_scale(),
                    zero_point=value.q_zero_point(),
                )
            return result
        if isinstance(value, (tuple, list)):
            return [_ExternalH800Baseline._input_description(x) for x in value]
        if isinstance(value, dict):
            return {
                k: _ExternalH800Baseline._input_description(v) for k, v in value.items()
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

        data = json.loads(payload, object_pairs_hook=unique, parse_constant=finite)
        aliases = ("_flash_attention_forward",)
        baseline_operator = next((name for name in aliases if name in data), None)
        if baseline_operator is None:
            raise ValueError(
                "Expected the canonical five-output ATen flash API baseline"
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
            if args[3] is not None:
                for tensor, total, maximum in (
                    (args[3], args[0].shape[0], args[5]),
                    (args[4], args[1].shape[0], args[6]),
                ):
                    if (
                        tensor.dtype != torch.int32
                        or tensor.device != args[0].device
                        or tensor.cpu().tolist() != [0, total - maximum, total]
                    ):
                        raise ValueError(
                            "Varlen cumulative lengths must match the recorded two-sequence input recipe"
                        )
            expected = {
                "args": self._input_description(args),
                "kwargs": self._input_description(kwargs),
            }
            shapes, provenance = self._h800_baseline
            matches = [
                record
                for dtypes in shapes.values()
                for record in dtypes.values()
                if record.get("config", {}).get("inputs") == expected
                and record.get("config", {}).get("case_id") == self._external_case_id
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


class FlashAttentionForwardBenchmark(_ExternalH800Baseline, OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "layout and FlashAttention dimensions/options"


def _case_fn(shape, dtype):
    del dtype
    mode = shape[0]
    if mode == "dense":
        (
            _,
            batch,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        tensor_shapes = {
            "query": [batch, q_len, q_heads, head_dim],
            "key": [batch, kv_len, kv_heads, head_dim],
            "value": [batch, kv_len, kv_heads, head_dim],
        }
    else:
        (
            _,
            sequences,
            heads,
            total_q,
            total_k,
            max_q,
            max_k,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        tensor_shapes = {
            "query": [total_q, heads, head_dim],
            "key": [total_k, heads, head_dim],
            "value": [total_k, heads, head_dim],
            "sequences": sequences,
            "max_q": max_q,
            "max_k": max_k,
        }
    yield base.BenchmarkCasePlan(
        shape=tensor_shapes,
        params={
            "mode": mode,
            "is_causal": is_causal,
            "dropout_p": dropout_p,
            "return_debug_mask": return_debug_mask,
            "window_size_left": window_left,
            "window_size_right": window_right,
        },
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    mode = shape[0]
    torch.manual_seed(0)
    if mode == "dense":
        (
            _,
            batch,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        query = torch.empty(
            (batch, q_len, q_heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        key = torch.empty(
            (batch, kv_len, kv_heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        value = torch.empty_like(key).uniform_(-0.05, 0.05)
        cum_seq_q = None
        cum_seq_k = None
        max_q = q_len
        max_k = kv_len
    else:
        (
            _,
            sequences,
            heads,
            total_q,
            total_k,
            max_q,
            max_k,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        query = torch.empty(
            (total_q, heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        key = torch.empty(
            (total_k, heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        value = torch.empty_like(key).uniform_(-0.05, 0.05)
        if sequences != 2:
            raise ValueError("the static varlen benchmark plan expects two sequences")
        cum_seq_q = torch.tensor(
            [0, total_q - max_q, total_q], dtype=torch.int32, device=device
        )
        cum_seq_k = torch.tensor(
            [0, total_k - max_k, total_k], dtype=torch.int32, device=device
        )
    kwargs = {"scale": 1.0 / math.sqrt(head_dim)}
    if window_left is not None or window_right is not None:
        kwargs["window_size_left"] = window_left
        kwargs["window_size_right"] = window_right
    return (
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        kwargs,
    )


@pytest.mark.underscore_flash_attention_forward
def test__flash_attention_forward():
    FlashAttentionForwardBenchmark(
        op_name="_flash_attention_forward",
        torch_op=torch.ops.aten._flash_attention_forward.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.bfloat16],
    ).run()
