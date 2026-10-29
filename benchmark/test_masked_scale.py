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

_MASKED_H800_SHA256 = "a8852fbf53b80d377b0565e9f0c087535708505e9fb23ba50eedb1b18b638769"
_MASKED_H800_SOURCE = "e713130f4a07162811d8dbd950c594be18a8d5f1"
_MASKED_CORE_SHAPES = [
    [1073741824],
    [64, 64],
    [4096, 4096],
    [64, 512, 512],
    [1024, 1024, 1024],
]


def _masked_h800_config(binding, dtype):
    shape = list(binding["shape"])
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * max(shape[i + 1], 1)

    def tensor(kind):
        return {
            "shape": shape,
            "stride": stride,
            "storage_offset": 0,
            "dtype": kind,
            "requires_grad": False,
        }

    return {
        "native_expression": "torch.ops.aten._masked_scale.default",
        "native_schema": "aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor",
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "source_benchmark_sha256": "b6227b7d29e0e871ec1d8a63b88413e92dbe42bd662bb5e80c6ee0c64eccba62",
        "source_builder_ast_sha256": "1e3dafe4b321481ab866cceac636bb52a0fcf0dc76d77d119fcb7b7395eaea2b",
        "source_tensor_factory_sha256": "2fa295e0f5d02e000d9e30967354cc663011de77366dd469feb8ddee7395a4a5",
        "benchmark": "benchmark/test_masked_scale.py",
        "builder": "MaskedScaleBenchmark.build_inputs",
        "scope": "core",
        "descriptor": shape,
        "variant": 0,
        "recipe": {
            "input": "torch.randn",
            "mask": "torch.randint(0, 2, dtype=torch.uint8)",
            "seed": 0,
            "on_requested_device": True,
        },
        "inputs": {
            "args": [tensor(str(dtype)), tensor("torch.uint8"), binding["scale"]],
            "kwargs": {},
        },
        "outputs": [{"name": "out", "shape": shape, "dtype": str(dtype)}],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }


def _masked_file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _MaskedH800Reference:
    def _require_masked_static(self, args, kwargs, case_id):
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError(
                "S60 masked_scale requires the exact native H800 --base-data"
            )
        if (
            self.op_name != "_masked_scale"
            or self.torch_op is not torch.ops.aten._masked_scale
        ):
            raise ValueError("Original masked_scale native callable/operator required")
        if (
            base.Config.bench_level != consts.BenchLevel.CORE
            or getattr(self, "fresh_inputs", False)
            or self.is_backward
        ):
            raise ValueError("Original forward shared-input masked_scale core required")
        if len(args) != 3 or kwargs:
            raise ValueError("Original input/mask/scale positional ABI required")
        inp, mask, scale = args
        target = torch.device(self.device)
        if target.index is None:
            target = torch.device(target.type, torch_device_fn.current_device())
        shape = list(inp.shape)
        if (
            shape not in _MASKED_CORE_SHAPES
            or inp.dtype != torch.float32
            or mask.dtype != torch.uint8
            or type(scale) is not float
            or scale != 2.0
        ):
            raise ValueError("Original masked_scale core dtype/shape/scale required")
        for tensor in (inp, mask):
            if (
                list(tensor.shape) != shape
                or tensor.device != target
                or tensor.storage_offset() != 0
                or not tensor.is_contiguous()
                or tensor.requires_grad
            ):
                raise ValueError("Original separate contiguous target inputs required")
        if inp.untyped_storage().data_ptr() == mask.untyped_storage().data_ptr():
            raise ValueError("Original input and mask must not alias")
        expected_id = (
            "benchmark/test_masked_scale.py::test_masked_scale::core::float32::"
            + str(_MASKED_CORE_SHAPES.index(shape))
        )
        if case_id != expected_id:
            raise ValueError("Original ordered masked_scale core identity required")
        baseline = base.Config.base_data
        if _masked_file_sha(baseline.source) != _MASKED_H800_SHA256:
            raise ValueError("Exact reviewed native H800 artifact required")
        data = baseline.data
        task = data["_collection"]["tasks"]["_masked_scale"]
        env = data["_environments"][str(task["device"])]
        if (
            data["_masked_scale"]["native_api"] != "ops._masked_scale"
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != _MASKED_H800_SOURCE
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
        ):
            raise ValueError("Completed clean native H800 masked_scale source required")
        call = describe_call(args, kwargs)
        expected = _masked_h800_config({"shape": shape, "scale": scale}, inp.dtype)
        expected_call = {
            "args": [
                (
                    {k: v for k, v in x.items() if k != "storage_offset"}
                    if isinstance(x, dict)
                    else x
                )
                for x in expected["inputs"]["args"]
            ],
            "kwargs": {},
        }
        record, key, reason = baseline.lookup(
            "_masked_scale", inp.dtype, call, target_device=self.device
        )
        if (
            record is None
            or reason is not None
            or record["config"] != expected
            or call != expected_call
            or not positive_number(record.get("T_us"))
            or not positive_number(record.get("latency_ms"))
            or not math.isclose(
                record["T_us"], 1000 * record["latency_ms"], rel_tol=1e-12
            )
        ):
            raise ValueError(
                "Actual masked_scale call must select its exact native H800 record"
            )
        return baseline, json.dumps(call, sort_keys=True), record["latency_ms"], key

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "enflame" and op is self.torch_op:
            raise ValueError(
                "S60 masked_scale requires an explicit supported candidate"
            )
        return op, dispatch, overridden

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name == "enflame":
            self._masked_checked = self._require_masked_static(
                *self.unpack_to_args_kwargs(input), case_id
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if flag_gems.vendor_name != "enflame":
            return super()._apply_external_baseline(metric, dtype, context)
        baseline, captured, latency, key = self._masked_checked
        if (
            context is None
            or context[0] is not baseline
            or base.Config.base_data is not baseline
            or json.dumps(context[1], sort_keys=True) != captured
            or _masked_file_sha(baseline.source) != _MASKED_H800_SHA256
        ):
            raise ValueError(
                "Native H800 masked_scale source/call changed around candidate timer"
            )
        super()._apply_external_baseline(metric, dtype, context)
        if (
            metric.latency == 0
            and metric.base_miss_reason == "invalid_candidate_latency"
        ):
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
            raise ValueError("Native H800 masked_scale baseline consumption failed")


class MaskedScaleBenchmark(_MaskedH800Reference, base.Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(dims), "mask": list(dims)},
                    params={"scale": 2.0},
                    builder_args=(dims,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        torch.manual_seed(0)
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        if flag_gems.vendor_name == "cambricon":
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device="cpu").to(
                self.device
            )
        else:
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=self.device)
        return inp, mask, plan.params["scale"]

    def get_tflops(self, op, *args, **kwargs):
        return args[0].numel()


@pytest.mark.masked_scale
@pytest.mark.underscore_masked_scale
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="S60 has no native masked_scale reference; consume exact native H800 core records",
)
def test_masked_scale():
    bench = MaskedScaleBenchmark(
        op_name="_masked_scale",
        torch_op=torch.ops.aten._masked_scale,
        dtypes=[torch.float32],
    )
    bench.run()
