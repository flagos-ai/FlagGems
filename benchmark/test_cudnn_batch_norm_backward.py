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

SHAPES = [(16, 16, 64), (32, 64, 56, 56)]


_H800_SHA = "a4d208fd94e726bcbdcea7b247d80707d30f57ba4747b4a60d2e79454a776740"
_H800_SOURCE = "62aa05b6a50a4ca4610fbadaba347efcb4682ebf"
_COLLECTION_OP = "cudnn_batch_norm_backward_current_core"
_PROVEN_CUDA_VERSIONS = {"2.8.0a0+5228986c39.nv25.05", "2.8.0a0+5228986c39.nv25.06"}


def _original_scope():
    return [dict(case_id="benchmark/test_cudnn_batch_norm_backward.py::test_cudnn_batch_norm_backward::core::"
                         + dtype.removeprefix("torch.") + "::" + str(i),
                 dtype=dtype, shape={"input": list(shape)}, params={})
            for dtype in ("torch.float16", "torch.float32", "torch.bfloat16")
            for i, shape in enumerate(SHAPES)]


def _read_h800():
    path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _H800_SHA:
        raise ValueError("Expected the reviewed six-scope native H800 BN snapshot")
    data = json.loads(payload)
    task = data["_collection"]["tasks"][_COLLECTION_OP]
    environment = data["_environments"][str(task["device"])]
    if (task["status"] != "completed" or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _H800_SOURCE
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["ncu_enabled"] is not False
            or data[_COLLECTION_OP]["native_api"] != "ops." + _COLLECTION_OP):
        raise ValueError("Incomplete clean native H800 BN provenance")
    expected = _original_scope()
    scope = data["_reference_scope"]
    unsupported = scope["unsupported_cases"]
    if (scope["native_torch_op"] != "aten::cudnn_batch_norm_backward.default"
            or scope["original_case_count"] != 6 or scope["cases"] != expected
            or scope["supported_case_ids"] != [x["case_id"] for x in expected[:4]]
            or [x["case"] for x in unsupported] != expected[4:]):
        raise ValueError("Native BN original six-case/unsupported identity mismatch")
    proof = data["_native_capability_evidence"]
    current = proof["gpu5_original_six_controls"]
    if (current["physical_device_index"] != int(task["device"])
            or current["hardware"]["uuid"] != environment["hardware"]["gpu"]["uuid"]
            or len(current["supported_controls"]) != 4
            or len(current["unsupported_controls"]) != 2):
        raise ValueError("Missing same-device native capability controls")
    variants = proof["gpu3_dtype_variants"]["rows"]
    for item in unsupported:
        native = item["native_probe"]
        shape = item["case"]["shape"]["input"]
        if (native["shape"] != shape or native["dtype"] != "torch.bfloat16"
                or native["operator"] != "aten::cudnn_batch_norm_backward.default"
                or native["status"] != "NATIVE_EXCEPTION" or native["error"] != "RuntimeError"
                or "BFloat16" not in native["message"]):
            raise ValueError("Unsupported BN identity lacks an actual matching native failure")
        matched = [x for x in variants if x["shape"] == shape]
        if (len(matched) != 4 or any(x["status"] != "NATIVE_EXCEPTION" for x in matched)
                or {(x["operator"], x["weight_statistics_dtype"]) for x in matched}
                != {(op, dtype) for op in ("forward", "backward")
                    for dtype in ("torch.float32", "torch.bfloat16")}):
            raise ValueError("Missing original BF16 same/mixed dtype native controls")
    records = data[_COLLECTION_OP]["shapes"]
    if sum(len(ds) for ds in records.values()) != 4 or any("torch.bfloat16" in ds for ds in records.values()):
        raise ValueError("Expected four measurements and no invented BF16 latency")
    return records, dict(kind="external_nvidia", artifact=str(path), artifact_sha256=_H800_SHA,
                         collection_task=task, device_environment=environment,
                         reference_scope=scope, ratio="H800 native cuDNN latency_ms / target candidate latency_ms"), scope


@dataclass
class _ExternalMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class OperatorCaseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "operator-specific shape"


    @cached_property
    def _external_records(self):
        records, source, _ = _read_h800()
        return records, source

    def _import_latency(self, args, kwargs):
        if len(args) != 9 or kwargs:
            raise ValueError("Expected original cuDNN backward nine-input ABI")
        inp, grad, weight, rm, rv, mean, invstd, eps, reserve = args
        channels = inp.shape[1]
        if (rm is not None or rv is not None or eps != 1e-5
                or inp.dtype not in (torch.float16, torch.float32)
                or grad.shape != inp.shape or grad.dtype != inp.dtype
                or any(x.shape != (channels,) or x.dtype != torch.float32 for x in (weight, mean, invstd))
                or reserve.shape != (0,) or reserve.dtype != torch.uint8
                or any(x.device != inp.device or not x.is_contiguous() or x.requires_grad
                       for x in (inp, grad, weight, mean, invstd, reserve))):
            raise ValueError("External baseline requires the corrected original supported BN recipe")
        def describe(x):
            return (dict(shape=list(x.shape), dtype=str(x.dtype), stride=list(x.stride()),
                         requires_grad=x.requires_grad) if isinstance(x, torch.Tensor) else x)
        config = dict(inputs=dict(args=[describe(x) for x in args], kwargs={}),
                      source_commit="882edec4093e02573b7bb274c76cbf9352e96ba8",
                      benchmark="benchmark/test_cudnn_batch_norm_backward.py", scope="core",
                      descriptor=list(inp.shape), variant=0,
                      input_recipe=dict(seed=0,input="torch.randn",grad="torch.randn",weight="ones_fp32",
                                        save_mean="randn_fp32",save_invstd="rand_fp32_plus_0.1",
                                        reserve="empty_uint8",eps=1e-5))
        records, source = self._external_records
        matches = [row for ds in records.values() for dtype, row in ds.items()
                   if dtype == str(inp.dtype) and row.get("config") == config]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous H800 record for full actual typed BN configuration")
        record = matches[0]
        latency = record["T_us"]
        if (isinstance(latency, bool) or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency <= 0
                or not math.isclose(record["latency_ms"], latency/1000, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError("Invalid H800 native latency or microsecond/millisecond units")
        self._baseline_source = dict(**source, config=config,
                                    target_environment=dict(vendor=flag_gems.vendor_name,
                                                            device=str(inp.device),torch=str(torch.__version__)))
        return latency / 1000

    def get_latency(self, op, *args, **kwargs):
            external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
            if getattr(self, "_import_reference_pending", False) or (external and base.Config.reference_only):
                self._import_reference_pending = False
                if op is not self.torch_op:
                    raise ValueError("Static H800 data may replace only the reference timer")
                return self._import_latency(args, kwargs)
            return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
            external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
            if flag_gems.vendor_name == "enflame" and not external:
                raise ValueError("Enflame BN timing requires static H800 native data")
            if external and (self.is_backward or base.Config.skip_native):
                raise ValueError("External H800 data requires native/candidate latency mode")
            self._baseline_source = None
            self._import_reference_pending = external and "latency_base" in self.to_bench_metrics
            try:
                metric = super()._measure_input(input, case_id)
                if self._baseline_source is not None:
                    return _ExternalMetrics(**asdict(metric), baseline_source=self._baseline_source)
                return metric
            finally:
                self._import_reference_pending = False

def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # Only these two original performance identities have proven unavailable
    # native H800 references. Unknown/missing identities are never skipped.
    if dtype == torch.bfloat16 and tuple(shape) in SHAPES:
        if os.environ.get("FLAGGEMS_NV_BASELINE"):
            _, _, scope = _read_h800()
            matches = [x for x in scope["unsupported_cases"] if x["case"]["shape"]["input"] == list(shape)]
            if len(matches) != 1:
                raise ValueError("Unsupported reference identity is not uniquely proven")
            pytest.skip(matches[0]["reason"])
        if flag_gems.vendor_name == "nvidia" and str(torch.__version__) in _PROVEN_CUDA_VERSIONS:
            pytest.skip("Matching native cuDNN reference is unavailable at the validated BF16 runtime")
    torch.manual_seed(0)
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    weight = torch.ones(channels, dtype=param_dtype, device=device)
    save_mean = torch.randn(channels, dtype=param_dtype, device=device)
    save_var = torch.rand(channels, dtype=param_dtype, device=device) + 0.1
    reserve = torch.empty((0,), dtype=torch.uint8, device=device)
    return inp, grad, weight, None, None, save_mean, save_var, 1e-5, reserve


@pytest.mark.cudnn_batch_norm_backward
def test_cudnn_batch_norm_backward():
    OperatorCaseBenchmark(
        op_name="cudnn_batch_norm_backward",
        torch_op=torch.ops.aten.cudnn_batch_norm_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
