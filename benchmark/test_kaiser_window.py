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


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"periodic": True, "beta": 12.0},
        builder_args=(shape, True),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"periodic": False, "beta": 12.0},
            builder_args=(shape, False),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, periodic = plan.builder_args
    return (
        {
            "window_length": math.prod(shape),
            "periodic": periodic,
            "beta": 12.0,
            "dtype": dtype,
            "device": device,
        },
    )


@dataclass
class _KaiserNativeMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _KaiserStaticBenchmark(base.GenericBenchmark):
    """Only the original plain Kaiser core; import the native reference timer."""

    @cached_property
    def _native_records(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        raw = path.read_bytes()
        digest = "7cfcf7426f705de7b52371f9ca7818a7201a4a82427cf6e03105d055021cb99e"
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Expected exact reviewed native Kaiser artifact")
        data = json.loads(raw)
        key = "kaiser_window_current_core"
        task = data["_collection"]["tasks"][key]
        env = data["_environment"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["hardware"]["gpu"]["uuid"] != "555633f8-c39c-bb8a-d549-791cb8451f44"
            or env["source"]["git_commit"] != "3975e9fa30eb83a5d2889ddc6baaca8fb29406f7"
            or env["source"]["git_dirty"] is not False
            or env["run"]["only"] != [key]
            or env["run"]["selected_ops"] != [key]
            or env["run"]["ncu_enabled"] is not False
            or data[key]["native_api"] != "ops." + key
        ):
            raise ValueError("Incomplete exact native Kaiser provenance")
        records = [r for ds in data[key]["shapes"].values() for r in ds.values()]
        if len(records) != 15:
            raise ValueError("Incomplete original Kaiser grid")
        return records, dict(
            kind="external_nvidia",
            artifact=str(path),
            artifact_sha256=digest,
            environment=env,
            collection_task=task,
            ratio="H800 native latency_ms / target candidate latency_ms",
        )

    def _import_latency(self, args, kwargs):
        if args or set(kwargs) != {
            "window_length",
            "periodic",
            "beta",
            "dtype",
            "device",
        }:
            raise ValueError("Expected original Kaiser factory keyword ABI")
        case_id = getattr(self, "_native_case_id", None)
        if base.Config.reference_only:
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise ValueError("Cannot identify exact original Kaiser case")
            case_id = active[0]["case_id"]
        cases = [c for c in self._collect_cases() if c.case_id == case_id]
        if len(cases) != 1:
            raise ValueError("Missing unique original factory case identity")
        case = cases[0]
        descriptor = list(case.shape["input"])
        shapes = [
            [1073741824],
            [64, 64],
            [4096, 4096],
            [64, 512, 512],
            [1024, 1024, 1024],
        ]
        if (
            descriptor not in shapes
            or case.params != {"periodic": True, "beta": 12.0}
            or kwargs["window_length"] != math.prod(descriptor)
            or type(kwargs["window_length"]) is not int
            or kwargs["periodic"] is not True
            or kwargs["beta"] != 12.0
            or type(kwargs["beta"]) is not float
            or kwargs["dtype"] != case.dtype
            or torch.empty(0, device=kwargs["device"]).device
            != torch.empty(0, device=self.device).device
        ):
            raise ValueError("Actual factory arguments differ from original core")
        config = dict(
            source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
            benchmark="benchmark/test_kaiser_window.py",
            scope="core",
            variant=0,
            descriptor=descriptor,
            native_api="torch.kaiser_window",
            inputs=dict(
                args=[],
                kwargs=dict(
                    window_length=math.prod(descriptor),
                    periodic=True,
                    beta=12.0,
                    dtype=str(case.dtype),
                    device="target",
                ),
            ),
            input_recipe="Exact original _build_inputs_fn; factory has no input tensor/RNG",
            output=dict(
                shape=[math.prod(descriptor)],
                dtype=str(case.dtype),
                device="target",
                alias=False,
                mutation=False,
            ),
        )
        records, source = self._native_records
        matches = [r for r in records if r.get("config") == config]
        if len(matches) != 1:
            raise ValueError("Missing unique native full factory case")
        record = matches[0]
        us, ms = record["T_us"], record["latency_ms"]
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in [us, ms]
        ) or not math.isclose(ms, us / 1000, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Invalid native Kaiser latency units")
        self._baseline_source = dict(
            source,
            config=config,
            case_id=case_id,
            target_candidate=(
                "CPU native semantic bootstrap restored to real target"
                if flag_gems.vendor_name == "enflame"
                else "native target"
            ),
        )
        if base.Config.reference_only:
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
            raise ValueError("Missing native H800 Kaiser reference")
        if external and (self.is_backward or base.Config.skip_native):
            raise ValueError("Expected native/candidate latency mode")
        self._native_case_id = case_id
        self._baseline_source = None
        self._native_pending = external and "latency_base" in self.to_bench_metrics
        try:
            metric = super()._measure_input(input, case_id)
            if self._baseline_source is not None:
                return _KaiserNativeMetrics(
                    **asdict(metric), baseline_source=self._baseline_source
                )
            return metric
        finally:
            self._native_pending = False
            self._native_case_id = None


@pytest.mark.kaiser_window
def test_kaiser_window():
    bench = _KaiserStaticBenchmark(
        op_name="kaiser_window",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.kaiser_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.kaiser_window_beta
def test_kaiser_window_beta():
    bench = base.GenericBenchmark(
        op_name="kaiser_window_beta",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.kaiser_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
