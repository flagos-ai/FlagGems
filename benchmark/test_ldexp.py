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

_H800_SHA256 = "d40a2eee1084c880663b207592d50934a32a85a6036405b4f18bca226a6bacd0"
_PUBLISHED_COMMIT = "60933ace321f266175a3aef14365db2302b7d4d5"
_COLLECTOR_COMMIT = "ccc0508d81b3b10283a62dab73fc05e256093848"
_BENCHMARK_COMMIT = "9d57a23b4386bd669152a935fe88a96db907da7c"
_CURRENT_CORE_SOURCE = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
_SUPPLEMENT_SOURCE = "ce1585f1dd7308bf82f42e4b084ead74b8f259e0"
_CURRENT_EXTRA_SHAPES = {(1073741824,), (64, 64), (64, 512, 512), (1024, 1024, 1024)}


@dataclass
class _ExternalMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


def _cpu_reference(inp, exponent, *, out=None):
    value = torch.ops.aten.ldexp.Tensor(inp.to("cpu"), exponent.to("cpu"))
    if out is not None:
        out.copy_(value.to(out.device))
        return out
    return value.to(inp.device)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={"exponent_dtype": "int32"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    return inp, exponent


class LdexpBenchmark(base.BinaryPointwiseBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    @cached_property
    def _external_records(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _H800_SHA256:
            raise ValueError("Expected the reviewed ops-baseline H800 snapshot")
        data = json.loads(payload)
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _COLLECTOR_COMMIT
            or environment["source"]["git_dirty"] is not False
            or data[self.op_name]["native_api"] != "ops." + self.op_name
        ):
            raise ValueError("Incomplete H800 native collection provenance")
        records = data[self.op_name]["shapes"]
        if sum(len(ds) for ds in records.values()) != 27:
            raise ValueError(
                "Expected legacy fifteen plus twelve current-core H800 records"
            )
        return records, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=_H800_SHA256,
            published_revision=_PUBLISHED_COMMIT,
            collection_task=task,
            device_environment=environment,
            ratio="H800 native latency_ms / target candidate latency_ms",
        )

    def _import_latency(self, args, kwargs):
        if len(args) != 2 or set(kwargs) != (
            {"out"} if self.op_name == "ldexp_out" else set()
        ):
            raise ValueError("Expected the original ldexp positional/out ABI")
        inp, exponent = args
        if (
            not isinstance(inp, torch.Tensor)
            or not isinstance(exponent, torch.Tensor)
            or inp.shape != exponent.shape
            or exponent.dtype != torch.int32
            or not inp.is_contiguous()
            or not exponent.is_contiguous()
            or inp.requires_grad
            or exponent.requires_grad
            or inp.device != exponent.device
        ):
            raise ValueError(
                "External baseline requires original contiguous input recipe"
            )
        # Reject input value distributions outside the original builder recipe.
        if exponent.numel() and (int(exponent.min()) < -8 or int(exponent.max()) > 8):
            raise ValueError("Exponent values differ from the original [-8,8] grid")
        describe = lambda x: dict(
            shape=list(x.shape),
            dtype=str(x.dtype),
            stride=list(x.stride()),
            requires_grad=x.requires_grad,
        )
        if kwargs:
            out = kwargs["out"]
            if (
                not isinstance(out, torch.Tensor)
                or out.shape != inp.shape
                or out.dtype != inp.dtype
                or out.device != inp.device
                or not out.is_contiguous()
                or out.requires_grad
            ):
                raise ValueError(
                    "External baseline requires the original dense out buffer"
                )
        config = dict(
            inputs=dict(
                args=[describe(inp), describe(exponent)],
                kwargs={k: describe(v) for k, v in kwargs.items()},
            ),
            source_commit=(
                _CURRENT_CORE_SOURCE
                if tuple(inp.shape) in _CURRENT_EXTRA_SHAPES
                else _BENCHMARK_COMMIT
            ),
            benchmark="benchmark/test_ldexp.py",
            descriptor=list(inp.shape),
            scope="core",
            variant=0,
        )
        if tuple(inp.shape) in _CURRENT_EXTRA_SHAPES:
            config["input_recipe"] = {
                "seed": 0,
                "input": "torch.randn",
                "exponent": "torch.randint(-8,9,int32)",
            }
        records, source = self._external_records
        matches = [
            record
            for ds in records.values()
            for dtype, record in ds.items()
            if dtype == str(inp.dtype) and record.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one H800 record matching the complete actual input config"
            )
        record = matches[0]
        origin = record.get("collection_origin")
        if tuple(inp.shape) in _CURRENT_EXTRA_SHAPES:
            if not isinstance(origin, dict):
                raise ValueError(
                    "Current-core record lacks original collection provenance"
                )
            environment = origin["environment"]
            if (
                environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
                or environment["source"]["git_commit"] != _SUPPLEMENT_SOURCE
                or environment["source"]["git_dirty"] is not False
                or environment["run"]["ncu_enabled"] is not False
                or origin["task"]["status"] != "completed"
                or origin["task"]["returncode"] != 0
            ):
                raise ValueError(
                    "Current-core data must come from the clean H800 supplement"
                )
            source = {
                **source,
                "collection_task": origin["task"],
                "device_environment": environment,
                "collection_origin": origin,
            }
        elif origin is not None:
            raise ValueError(
                "Unexpected replacement provenance for a preserved legacy record"
            )
        latency = record["T_us"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError("H800 latency must be finite positive microseconds")
        milliseconds = latency / 1000
        if not math.isclose(
            record["latency_ms"], milliseconds, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("Inconsistent H800 latency units")
        self._baseline_source = {
            **source,
            "config": config,
            "target_environment": dict(
                vendor=flag_gems.vendor_name,
                device=str(inp.device),
                torch=str(torch.__version__),
            ),
            "input_recipe": dict(
                input="torch.randn", exponent="torch.randint(-8,9,int32)"
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
                raise RuntimeError("Cannot identify active external baseline case")
            active[0]["baseline_source"] = self._baseline_source
        return milliseconds

    def get_latency(self, op, *args, **kwargs):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        reference_only = base.Config.reference_only
        if reference_only and flag_gems.vendor_name == "enflame" and not external:
            raise ValueError(
                "Enflame reference timing requires static H800 native data"
            )
        if getattr(self, "_import_reference_pending", False) or (
            external and reference_only
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
        if flag_gems.vendor_name == "enflame" and not external:
            raise ValueError(
                "Enflame ldexp requires FLAGGEMS_NV_BASELINE H800 native data"
            )
        if external and (self.is_backward or base.Config.skip_native):
            raise ValueError(
                "External H800 data requires forward native/candidate latency mode"
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

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.ldexp
def test_ldexp():
    bench = LdexpBenchmark(
        op_name="ldexp",
        torch_op=(
            _cpu_reference
            if flag_gems.vendor_name == "enflame"
            else torch.ops.aten.ldexp.Tensor
        ),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _out_build_inputs_fn(plan, dtype, device):
    inp, exponent = _build_inputs_fn(plan, dtype, device)
    out = torch.empty_like(inp)
    return inp, exponent, {"out": out}


@pytest.mark.ldexp_out
def test_ldexp_out():
    LdexpBenchmark(
        op_name="ldexp_out",
        torch_op=(
            _cpu_reference
            if flag_gems.vendor_name == "enflame"
            else torch.ops.aten.ldexp.out
        ),
        case_fn=_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    ).run()
