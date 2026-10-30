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


@dataclass
class _LdexpNativeMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _LdexpStaticBaseline:
    """Only this original in-place core imports its fresh native reference timer."""

    @cached_property
    def _native_records(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        raw = path.read_bytes()
        digest = "cde9d47c4da7478951adc6f90013e14575dd561409a4b77b5e5160992c15865f"
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Expected exact reviewed fresh-native ldexp artifact")
        data = json.loads(raw)
        key = "ldexp_inplace_current_core"
        env, method, task = (
            data["_environment"],
            data["_measurement_method"],
            data["_execution"],
        )
        expected_method = (
            "official collector + operator-specific untimed-refresh CUDA-event timing"
        )
        if (
            task["exit"] != 0
            or task["source_before_after"] is not True
            or task["head"] != "0ca0827b1440a2a20b717c844966205fbe2a44b2"
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["hardware"]["gpu"]["uuid"] != "f76856dc-d02f-95a0-dc05-054190d1ea7d"
            or env["source"]["git_commit"] != task["head"]
            or env["source"]["git_dirty"] is not False
            or env["run"]["only"] != [key]
            or env["run"]["selected_ops"] != [key]
            or env["run"]["ncu_enabled"] is not False
            or env["run"]["latency_method"] != expected_method
            or method["name"] != expected_method
            or method["ncu_profiled"] is not False
            or data[key]["native_api"] != "ops." + key
        ):
            raise ValueError("Incomplete exact fresh-native ldexp provenance")
        records = [r for ds in data[key]["shapes"].values() for r in ds.values()]
        if len(records) != 15 or any(
            r["measurement_method"]["name"] != expected_method
            or r["measurement_method"]["driver_sha256"] != method["driver_sha256"]
            for r in records
        ):
            raise ValueError("Incomplete original fresh-native grid or timing method")
        return records, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=digest,
            environment=env,
            execution=task,
            native_measurement_method=method,
            ratio="H800 fresh native event latency_ms / target candidate latency_ms",
        )

    def _import_latency(self, args, kwargs):
        if (
            kwargs
            or len(args) != 2
            or not all(isinstance(x, torch.Tensor) for x in args)
        ):
            raise ValueError("Expected original ldexp_ positional tensor ABI")
        inp, exponent = args
        shapes = [
            (1073741824,),
            (64, 64),
            (4096, 4096),
            (64, 512, 512),
            (1024, 1024, 1024),
        ]
        if (
            tuple(inp.shape) not in shapes
            or exponent.shape != inp.shape
            or inp.dtype not in [torch.float16, torch.float32, torch.bfloat16]
            or exponent.dtype != torch.int32
            or exponent.device != inp.device
            or inp.device != torch.empty(0, device=self.device).device
            or not inp.is_contiguous()
            or not exponent.is_contiguous()
            or inp.requires_grad
            or exponent.requires_grad
        ):
            raise ValueError("Actual original ldexp_ input metadata differ")
        describe = lambda x: dict(
            shape=list(x.shape),
            dtype=str(x.dtype),
            stride=list(x.stride()),
            requires_grad=False,
        )
        method = (
            "official collector + operator-specific untimed-refresh CUDA-event timing"
        )
        config = dict(
            source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
            benchmark="benchmark/test_ldexp_.py",
            scope="core",
            variant=0,
            descriptor=list(inp.shape),
            native_api="torch.ops.aten.ldexp_.default",
            inputs=dict(args=[describe(inp), describe(exponent)], kwargs={}),
            input_recipe=dict(
                rng="manual_seed(0)",
                self="randn",
                exponent="randint(-8,9,int32)",
                fresh_inputs=True,
            ),
            measurement_method=method,
            mutation="self only",
            output_alias="self",
        )
        records, source = self._native_records
        matches = [r for r in records if r.get("config") == config]
        if len(matches) != 1:
            raise ValueError("Missing unique native fresh-input ldexp case")
        record = matches[0]
        us, ms = record["T_us"], record["latency_ms"]
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in [us, ms]
        ) or not math.isclose(ms, us / 1000, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Invalid native ldexp latency units")
        self._baseline_source = dict(
            source,
            config=config,
            target_candidate=(
                "locally timed candidate; implementation supplied by the runner"
            ),
        )
        if base.Config.reference_only:
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise ValueError("Cannot identify active ldexp reference case")
            active[0]["baseline_source"] = self._baseline_source
        return ms

    def get_latency(self, op, *args, **kwargs):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if getattr(self, "_native_pending", False) or (
            external and base.Config.reference_only
        ):
            self._native_pending = False
            if op is not self.torch_op:
                raise ValueError("Native data replaces only reference timer")
            return self._import_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if flag_gems.vendor_name == "enflame" and not external:
            raise ValueError("Missing fresh native H800 ldexp reference")
        if external and (self.is_backward or base.Config.skip_native):
            raise ValueError("Expected native/candidate latency mode")
        self._baseline_source = None
        self._native_pending = external and "latency_base" in self.to_bench_metrics
        try:
            metric = super()._measure_input(input, case_id)
            if self._baseline_source is not None:
                return _LdexpNativeMetrics(
                    **asdict(metric), baseline_source=self._baseline_source
                )
            return metric
        finally:
            self._native_pending = False


class LdexpInplaceBenchmark(
    _LdexpStaticBaseline, base.BinaryPointwiseBenchmark, OperatorBenchmark
):
    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.ldexp_
def test_ldexp_():
    bench = LdexpInplaceBenchmark(
        op_name="ldexp_",
        torch_op=torch.ops.aten.ldexp_.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
