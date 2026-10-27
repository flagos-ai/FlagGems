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

SHAPES = [(16, 8, 64), (32, 16, 128), (64, 32, 256), (128, 64, 256)]


@dataclass
class _PackMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _PackBaseline:
    def _external_input_config(self, args):
        if len(args) != 3:
            raise ValueError("Pack baseline requires input, lengths, batch_first")
        inp, lengths, batch_first = args
        if (
            not isinstance(inp, torch.Tensor)
            or not isinstance(lengths, torch.Tensor)
            or inp.layout != torch.strided
            or lengths.layout != torch.strided
            or lengths.device.type != "cpu"
            or lengths.dtype != torch.int64
            or type(batch_first) is not bool
        ):
            raise ValueError(
                "Pack baseline requires dense input and native CPU lengths ABI"
            )

        def tensor(value):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "stride": list(value.stride()),
                "requires_grad": value.requires_grad,
            }

        return {
            "inputs": {
                "args": [
                    tensor(inp),
                    {**tensor(lengths), "device": "cpu", "values": lengths.tolist()},
                    batch_first,
                ],
                "kwargs": {},
            },
            "source_commit": "00cd80f89b12864b2e6cf711db283d3d95b2187a",
            "benchmark": "benchmark/test_pack_padded_sequence.py",
            "descriptor": list(inp.shape),
            "scope": "core",
            "variant": 0,
            "reference_implementation": "pytorch_native_operator",
        }

    @cached_property
    def _nvidia_baseline(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate baseline JSON key: " + key)
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError("Non-finite baseline JSON constant: " + value)

        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        key = "pack_padded_sequence"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        if (
            data[key]["native_api"] != "ops." + key
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
                "Pack baseline requires a completed, traceable native H800 run"
            )
        return data[key]["shapes"], {
            "kind": "external_nvidia",
            "operator": key,
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "environment": environment,
            "reference_implementation": "pytorch_native_operator",
            "input_identity": "same distribution and exact input contract; random payloads are not byte-identical",
            "ratio": "raw H800 latency / target latency; no peak normalization",
            "target_timing_mode": base.Config.mode.value,
        }

    def _import_nvidia_latency(self, args, kwargs):
        if kwargs or self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Pack external baseline supports the original positional core ABI only"
            )
        config = self._external_input_config(args)
        shapes, provenance = self._nvidia_baseline
        matches = [
            r
            for dtypes in shapes.values()
            for dtype, r in dtypes.items()
            if dtype == str(args[0].dtype) and r.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError("Expected one H800 record matching actual pack inputs")
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
                "device": str(args[0].device),
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
                    "Cannot identify active pack external reference case"
                )
            active[0]["baseline_source"] = self._imported_baseline
        return latency_ms

    def get_latency(self, op, *args, **kwargs):
        external = flag_gems.vendor_name == "ascend" and bool(
            os.environ.get("FLAGGEMS_NV_BASELINE")
        )
        if external and (
            base.Config.reference_only
            or getattr(self, "_native_measurement_pending", False)
        ):
            if base.Config.skip_native or op is not self.torch_op:
                raise ValueError(
                    "External baseline requires original reference measurement"
                )
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        if flag_gems.vendor_name == "ascend" and op is self.torch_op and not external:
            raise ValueError(
                "Ascend native pack has incorrect variable-length semantics; set FLAGGEMS_NV_BASELINE to the H800 artifact"
            )
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        external = flag_gems.vendor_name == "ascend" and bool(
            os.environ.get("FLAGGEMS_NV_BASELINE")
        )
        if external and (
            self.is_backward
            or "tflops" in self.to_bench_metrics
            or base.Config.skip_native
        ):
            raise ValueError(
                "Pack external baseline requires forward latency measurement without skip_native"
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
                return _PackMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def _run_reference_cases(self, case_ids):
        first_record = len(base.Config.reference_records)
        if (
            flag_gems.vendor_name == "ascend"
            and os.environ.get("FLAGGEMS_NV_BASELINE")
            and base.Config.skip_native
        ):
            raise ValueError("Do not combine an external baseline with skip_native")
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            for record in base.Config.reference_records[first_record:]:
                if record.get("baseline_source") and record.get("status") == "PASSED":
                    record["stage"] = "external_baseline"


class PackPaddedSequenceBenchmark(_PackBaseline, OperatorBenchmark):
    DEFAULT_SHAPES = SHAPES

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SHAPES)


def _case_fn(shape, dtype):
    del dtype
    time_steps, batch_size, features = shape
    yield base.BenchmarkCasePlan(
        shape={"input": [time_steps, batch_size, features], "lengths": [batch_size]},
        params={"batch_first": False},
        builder_args=(time_steps, batch_size, features),
    )


def _build_inputs_fn(plan, dtype, device):
    time_steps, batch_size, features = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(time_steps, batch_size, features, dtype=dtype, device=device)
    lengths = torch.linspace(time_steps, 1, steps=batch_size, dtype=torch.int64).clamp(
        min=1
    )
    lengths = torch.sort(lengths, descending=True).values
    return inp, lengths, False


@pytest.mark.pack_padded_sequence
def test_pack_padded_sequence():
    PackPaddedSequenceBenchmark(
        op_name="pack_padded_sequence",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._pack_padded_sequence,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
