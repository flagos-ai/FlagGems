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
from pathlib import Path

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base, consts, utils
from .baseline import describe_call, positive_number

_MSORT_H800_SHA256 = "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3"
_MSORT_NATIVE_LATENCIES = {
    "msort": {
        "[8, 1024]": {
            "torch.float16": [51.03251036811382, 0.05103251036811382],
            "torch.float32": [19.030697649171533, 0.01903069764917153],
            "torch.bfloat16": [32.24530326641245, 0.03224530326641245],
        },
        "[32, 4096]": {
            "torch.float16": [32.77360329916637, 0.03277360329916637],
            "torch.float32": [32.27643732974927, 0.03227643732974927],
            "torch.bfloat16": [23.41273486843728, 0.02341273486843728],
        },
        "[128, 8192]": {
            "torch.float16": [72.34071561942922, 0.07234071561942922],
            "torch.float32": [89.34966116810644, 0.08934966116810644],
            "torch.bfloat16": [72.52759378018919, 0.07252759378018919],
        },
        "[512, 4096]": {
            "torch.float16": [142.71818186071786, 0.14271818186071786],
            "torch.float32": [180.3652621447084, 0.18036526214470838],
            "torch.bfloat16": [142.34851810780597, 0.14234851810780597],
        },
        "[32, 131072]": {
            "torch.float16": [188.1260688620052, 0.18812606886200522],
            "torch.float32": [207.16645784047714, 0.20716645784047713],
            "torch.bfloat16": [188.8520675527174, 0.18885206755271738],
        },
    },
    "msort_out": {
        "[8, 1024]": {
            "torch.float16": [48.621599934995174, 0.04862159993499517],
            "torch.float32": [17.67025494066835, 0.01767025494066835],
            "torch.bfloat16": [27.572869573740046, 0.027572869573740045],
        },
        "[32, 4096]": {
            "torch.float16": [22.58773106039217, 0.02258773106039217],
            "torch.float32": [23.436707984326425, 0.023436707984326424],
            "torch.bfloat16": [24.06452749190586, 0.02406452749190586],
        },
        "[128, 8192]": {
            "torch.float16": [71.05883612082555, 0.07105883612082554],
            "torch.float32": [88.21074527718991, 0.08821074527718992],
            "torch.bfloat16": [71.2487257328354, 0.0712487257328354],
        },
        "[512, 4096]": {
            "torch.float16": [141.39692306653407, 0.14139692306653406],
            "torch.float32": [179.18760742527738, 0.17918760742527737],
            "torch.bfloat16": [141.1775127562412, 0.1411775127562412],
        },
        "[32, 131072]": {
            "torch.float16": [186.75752075617774, 0.18675752075617774],
            "torch.float32": [206.177080015829, 0.20617708001582902],
            "torch.bfloat16": [187.81821056139094, 0.18781821056139095],
        },
    },
}
_MSORT_CORE_SHAPES = [[8, 1024], [32, 4096], [128, 8192], [512, 4096], [32, 131072]]


def _msort_file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _MsortH800Reference:
    def _msort_static_applies(self):
        return flag_gems.vendor_name == "enflame"

    def _require_msort_static(self, args, kwargs, case_id):
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError("S60 msort requires the exact native H800 --base-data")
        if (
            self.op_name not in _MSORT_NATIVE_LATENCIES
            or self.torch_op is not torch.msort
        ):
            raise ValueError("Original native msort callable/operator required")
        if base.Config.bench_level != consts.BenchLevel.CORE or getattr(
            self, "fresh_inputs", False
        ):
            raise ValueError("Original shared-input core msort profile required")
        if len(args) != 1 or set(kwargs) != (
            {"out"} if self.op_name == "msort_out" else set()
        ):
            raise ValueError("Original msort input/out signature differs")
        inp = args[0]
        if (
            inp.dtype not in (torch.float16, torch.float32, torch.bfloat16)
            or list(inp.shape) not in _MSORT_CORE_SHAPES
        ):
            raise ValueError("Original msort core dtype/shape required")
        target = torch.device(self.device)
        if target.index is None:
            target = torch.device(target.type, torch_device_fn.current_device())
        expected_id = (
            "benchmark/test_"
            + self.op_name
            + ".py::test_"
            + self.op_name
            + "::core::"
            + str(inp.dtype).removeprefix("torch.")
            + "::"
            + str(_MSORT_CORE_SHAPES.index(list(inp.shape)))
        )
        if case_id != expected_id:
            raise ValueError("Original msort ordered core case identity required")
        if not inp.is_contiguous() or inp.requires_grad or inp.device != target:
            raise ValueError("Original contiguous target input required")
        if self.op_name == "msort_out":
            out = kwargs["out"]
            if (
                out.device != inp.device
                or out.dtype != inp.dtype
                or out.shape != inp.shape
                or out.stride() != inp.stride()
                or out.requires_grad
                or out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
            ):
                raise ValueError("Original separate contiguous msort out required")
        baseline = base.Config.base_data
        if _msort_file_sha(baseline.source) != _MSORT_H800_SHA256:
            raise ValueError("Exact published native H800 artifact required")
        task = baseline.data["_collection"]["tasks"][self.op_name]
        env = baseline.data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != "ccc0508d81b3b10283a62dab73fc05e256093848"
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
        ):
            raise ValueError("Completed clean native H800 msort source required")
        call = describe_call(args, kwargs)
        shape = list(inp.shape)
        tensor = dict(
            shape=shape, dtype=str(inp.dtype), stride=[shape[1], 1], requires_grad=False
        )
        expected = dict(
            inputs=dict(
                args=[tensor],
                kwargs={"out": dict(tensor)} if self.op_name == "msort_out" else {},
            ),
            source_commit="9d57a23b4386bd669152a935fe88a96db907da7c",
            benchmark="benchmark/test_msort.py",
            descriptor=shape,
            scope="core",
            variant=0,
        )
        record, key, reason = baseline.lookup(
            self.op_name, inp.dtype, call, target_device=self.device
        )
        tu, ms = _MSORT_NATIVE_LATENCIES[self.op_name][json.dumps(shape)][
            str(inp.dtype)
        ]
        if (
            record is None
            or reason is not None
            or record["config"] != expected
            or call != expected["inputs"]
            or record["T_us"] != tu
            or record["latency_ms"] != ms
            or not positive_number(ms)
            or not math.isclose(tu, 1000 * ms, rel_tol=1e-12)
        ):
            raise ValueError(
                "Actual msort call must select its exact native H800 record"
            )
        return baseline, json.dumps(call, sort_keys=True), ms, key

    def _measure_input(self, input, case_id=None):
        if self._msort_static_applies():
            self._msort_checked = self._require_msort_static(
                *self.unpack_to_args_kwargs(input), case_id
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if not self._msort_static_applies():
            return super()._apply_external_baseline(metric, dtype, context)
        baseline, captured, latency, key = self._msort_checked
        if (
            context is None
            or context[0] is not baseline
            or base.Config.base_data is not baseline
            or json.dumps(context[1], sort_keys=True) != captured
            or _msort_file_sha(baseline.source) != _MSORT_H800_SHA256
        ):
            raise ValueError(
                "Native H800 msort source/call changed around candidate timer"
            )
        super()._apply_external_baseline(metric, dtype, context)
        if (
            metric.latency == 0
            and metric.base_miss_reason == "invalid_candidate_latency"
        ):
            # Keep the real zero and missing ratio; the separately verified H800 record is still identified.
            metric.base_data_source, metric.base_shape_key, metric.base_latency_ms = (
                baseline.source,
                key,
                latency,
            )
            return
        if (
            metric.error_msg is not None
            or metric.latency_base is not None
            or metric.base_miss_reason is not None
            or metric.base_data_source != baseline.source
            or metric.base_shape_key != key
            or metric.base_latency_ms != latency
        ):
            raise ValueError("Native H800 msort baseline consumption failed")


class MsortBenchmark(_MsortH800Reference, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 1024),
            (32, 4096),
            (128, 8192),
            (512, 4096),
            (32, 131072),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(self.set_more_shapes())

    def set_more_shapes(self):
        return [(8, 262144), (128, 65536), (512, 16384), (32, 524288)]

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        return (utils.generate_tensor_input(shape, case.dtype, self.device),)


@pytest.mark.msort
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="S60 native msort aborts; consume exact native H800 core records",
)
def test_msort():
    MsortBenchmark(
        op_name="msort", torch_op=torch.msort, dtypes=consts.FLOAT_DTYPES
    ).run()
