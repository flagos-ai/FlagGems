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
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    if flag_gems.vendor_name == "iluvatar" and dtype == torch.float64:
        pytest.skip("Iluvatar Float64 native nextafter returns incorrect values")
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double nextafter values")
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        # Device randn narrows Double; retain the requested normal recipe and
        # move its bytes without invoking that narrowing conversion.
        host = [torch.randn(shape, dtype=dtype, device="cpu") for _ in range(2)]
        inputs = tuple(
            value.reshape(-1).view(torch.uint8).to(device).view(dtype).reshape(shape)
            for value in host
        )
        assert all(
            value.dtype == dtype
            and value.device.type == torch.device(device).type
            and value.element_size() == 8
            and value.untyped_storage().nbytes() == value.numel() * 8
            for value in inputs
        )
        return inputs
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape, dtype=dtype, device=device),
    )


@dataclass
class _NextafterMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate nextafter baseline JSON key: " + key)
        result[key] = value
    return result


class _AscendNextafterBaseline:
    @cached_property
    def _h800_records(self):
        path = Path(os.environ["FLAGGEMS_NEXTAFTER_H800_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "56a2f1398cd9ca0648dc2104053041e67ca2063dac5e643a35b574621ea474c4":
            raise ValueError("Unreviewed nextafter H800 artifact")
        data = json.loads(payload, object_pairs_hook=_unique_json)
        records = {}
        for origin, count in (("legacy", 15), ("own", 20)):
            key = "nextafter_inplace_" + origin + "_core"
            task = data["_collection"]["tasks"][key]
            environment = data["_environments"][str(task["device"])]
            if (
                task["status"] != "completed"
                or task["returncode"] != 0
                or data[key]["native_api"] != "ops." + key
                or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
                or environment["source"]["git_commit"]
                != "edcf1d0b609106402861d5c95c38534828de7763"
                or environment["source"]["git_dirty"] is not False
                or environment["run"]["repo"] != "pytorch"
                or environment["run"]["ncu_enabled"] is not False
            ):
                raise ValueError("Nextafter requires completed native H800 profiles")
            rows = [r for ds in data[key]["shapes"].values() for r in ds.values()]
            if len(rows) != count:
                raise ValueError("Incomplete nextafter H800 profile")
            for row in rows:
                case_id = row["config"]["original_case_id"]
                if case_id in records:
                    raise ValueError("Duplicate nextafter H800 case identity")
                records[case_id] = (row, environment)
        return records, path, digest

    def _h800_latency(self, inp, other):
        case = self._nextafter_case
        if (
            self.is_backward
            or base.Config.skip_native
            or inp.device.type != torch.device(self.device).type
            or other.device != inp.device
            or inp.layout != torch.strided
            or other.layout != torch.strided
        ):
            raise ValueError("Nextafter H800 baseline requires the original core ABI")
        case_file = case.case_id.split("::", 1)[0]
        if case_file not in {
            "benchmark/test_nextafter.py",
            "benchmark/test_nextafter_.py",
        }:
            raise ValueError("Unknown nextafter benchmark origin")
        origin = "legacy" if case_file.endswith("test_nextafter.py") else "own"
        shape = list(inp.shape)
        builders = (
            {
                "_legacy_case_fn": "d3c26b4539f52d505c7af05e8296fdc680de03bed0129156e143571ce775ba3c",
                "_legacy_build_inputs_fn": "e442184d59fa4a31a5c0a9631a653bb0b0650b785516f76b9c90d3e01a3fdba9",
            }
            if origin == "legacy"
            else {
                "_case_fn": "4d1e07d4be240475f1f4c1de7155ec73e431cb0a58242188a3f4fc90df789701",
                "_build_inputs_fn": "a99e2b5f5789019bd6bfbf7a1e4144d2f07d13f098b8c3b80bf18c428219e067",
            }
        )

        def tensor(value):
            return dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                stride=list(value.stride()),
                storage_offset=value.storage_offset(),
                requires_grad=value.requires_grad,
            )

        expected = dict(
            canonical_operator="nextafter_",
            native_expression="torch.Tensor.nextafter_",
            source_commit="c0c8740c0d312b153c92e30511b739f49988f226",
            benchmark=case_file,
            original_case_id=case.case_id,
            ordinal=case.ordinal,
            origin=origin,
            original_shape=case.shape,
            original_params=case.params,
            descriptor=shape,
            scope="core",
            builder_AST_SHA256=builders,
            inputs=dict(args=[tensor(inp), tensor(other)], kwargs={}),
            input_recipe=dict(
                generator="torch.randn",
                seed=None if origin == "legacy" else 0,
                dtype=str(inp.dtype),
                allocation_order=["self", "other"],
            ),
            mutation=dict(mutates=["self"], readonly=["other"], result_alias="self"),
            timing_protocol=dict(
                prepare_sample="clone self only; keep original other object",
                prepare_before_CUDA_start_event=True,
                native_callable="Tensor.nextafter_",
                warmup_ms=25,
                rep_ms=100,
                return_mode="mean",
                ncu=False,
            ),
        )
        records, path, digest = self._h800_records
        row, environment = records[case.case_id]
        # Normalize tuple/list representations, preserving every joint field.
        if json.loads(json.dumps(expected)) != row["config"]:
            raise ValueError("Nextafter H800 full configuration mismatch")
        latency = row["T_us"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
            or not math.isclose(
                row["latency_ms"], latency / 1000, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "Nextafter H800 latency units must be positive and consistent"
            )
        self._imported_nextafter = dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=digest,
            config=expected,
            environment=environment,
            environment_scope="artifact last-worker environment header; per-profile selected-op provenance is retained in the original worker results",
            input_identity="exact recipes and joint configuration; independent random payloads",
            target_timing_mode=base.Config.mode.value,
            target_implementation="injected/default candidate timed locally; implementation not inferred",
            ratio="raw H800 milliseconds / target milliseconds; no normalization",
        )
        return latency / 1000

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name != "ascend":
            return super()._measure_input(input, case_id)
        self._imported_nextafter = None
        metric = super()._measure_input(input, case_id)
        if self._imported_nextafter is not None:
            records, path, digest = self._h800_records
            del records
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Nextafter H800 artifact changed during measurement")
            return _NextafterMetrics(
                **asdict(metric), baseline_source=self._imported_nextafter
            )
        return metric


def _fresh_nextafter_latency(self, op, inp, other):
    if inp.untyped_storage()._cdata == other.untyped_storage()._cdata:
        raise ValueError("Fresh nextafter benchmark inputs must not alias")
    if flag_gems.vendor_name == "ascend" and op is self.torch_op:
        return self._h800_latency(inp, other)
    # Only self may change. Keep the independent read-only other tensor and
    # clone self through the existing timer hook, outside every timed call.
    return self._get_fresh_input_latency(
        lambda fresh_self: op(fresh_self, other), (inp,), {}
    )


class NextafterInplaceBenchmark(
    _AscendNextafterBaseline, base.BinaryPointwiseBenchmark, OperatorBenchmark
):
    get_latency = _fresh_nextafter_latency

    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter

    def build_inputs(self, case):
        if flag_gems.vendor_name == "ascend":
            self._nextafter_case = case
        if flag_gems.vendor_name == "kunlunxin" and case.dtype == torch.float16:
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("Kunlunxin native Half nextafter loses original-dtype ULPs")
        return OperatorBenchmark.build_inputs(self, case)


@pytest.mark.nextafter_
def test_nextafter_():
    bench = NextafterInplaceBenchmark(
        op_name="nextafter_",
        torch_op=lambda a, b: a.nextafter_(b),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32, torch.float64],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
