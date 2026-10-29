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
from pathlib import Path

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base, consts, utils
from .baseline import describe_call, positive_number
from .generated_operator_utils import OperatorBenchmark

GROUPED_MM_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


_GROUP_MM_H800_SHA256 = (
    "849482bde1986b10a7e51387b34cb41e3728d7f7d2cb427cd3dfb59ff60ca4eb"
)
_GROUP_MM_H800_SOURCE = "ab6d02fa3fb36fb42eac0783798ddff0e79be34c"


def _native_group_mm_composite(mat_a, mat_b, offs):
    assert mat_a.ndim == 2 and mat_b.ndim == 3 and offs.ndim == 1
    assert mat_a.dtype == mat_b.dtype == torch.bfloat16 and offs.dtype == torch.int32
    assert mat_a.device == mat_b.device == offs.device and mat_a.device.type == "cuda"
    assert mat_a.is_contiguous() and mat_b.is_contiguous() and offs.is_contiguous()
    assert not any(t.requires_grad for t in (mat_a, mat_b, offs))
    groups, k_size, _ = mat_b.shape
    assert mat_a.shape[1] == k_size and offs.numel() == groups
    boundaries = [0] + offs.cpu().tolist()
    assert boundaries[-1] == mat_a.shape[0]
    assert all(left < right for left, right in zip(boundaries, boundaries[1:]))
    return torch.cat(
        [
            torch.mm(mat_a[boundaries[i] : boundaries[i + 1]], mat_b[i])
            for i in range(groups)
        ],
        dim=0,
    )


def _sizes(binding):
    groups, _, _ = binding["shape"]
    return [16 + (idx * 13) % 48 for idx in range(groups)]


def _group_mm_h800_config(binding, dtype):
    groups, n_size, k_size = binding["shape"]
    sizes = _sizes(binding)
    m_size = sum(sizes)

    def tensor(shape, tensor_dtype, stride):
        return dict(
            shape=shape,
            dtype=str(tensor_dtype),
            stride=stride,
            storage_offset=0,
            requires_grad=False,
        )

    return dict(
        native_expression="native_CUDA_per_group_mm_actual_offsets_D2H_cat",
        literal_grouped_kernel=False,
        timed_components=[
            "offs.cpu().tolist",
            "Python group loop",
            "torch.mm each original group",
            "torch.cat",
        ],
        source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        source_benchmark_sha256="552ddc59212ff654de494630215c8958c83aebdd10bfb4bafdc411030c38a4dd",
        source_builder_ast_sha256="23cdc17663c803e4e87d8101cc68803571cb24f0a396c6939b1c0d9eceb6467d",
        benchmark="benchmark/test_grouped_mm.py",
        scope="core",
        descriptor=list(binding["shape"]),
        group_sizes=sizes,
        offsets=[sum(sizes[: i + 1]) for i in range(groups)],
        recipe=dict(
            A="torch.rand",
            B="torch.rand",
            offsets="torch.tensor cumulative endpoints",
            seed=0,
            original_dtype=str(dtype),
        ),
        inputs=dict(
            args=[
                tensor([m_size, k_size], dtype, [k_size, 1]),
                tensor([groups, k_size, n_size], dtype, [k_size * n_size, n_size, 1]),
                tensor([groups], torch.int32, [1]),
            ],
            kwargs={},
        ),
        output=tensor([m_size, n_size], dtype, [n_size, 1]),
        effects=dict(mutates=[], returns_alias_of={}),
    )


class _GroupMMH800Reference:
    def _require_composite_static(self, args, kwargs, case_id):
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError(
                "Selected backend group_mm requires exact native-composite H800 --base-data"
            )
        if (
            self.op_name != "group_mm"
            or self.torch_op is not _native_group_mm_composite
            or base.Config.bench_level != consts.BenchLevel.CORE
            or getattr(self, "fresh_inputs", False)
            or self.is_backward
        ):
            raise ValueError("Original forward shared-input group_mm core required")
        if len(args) != 3 or kwargs:
            raise ValueError("Original three positional inputs required")
        mat_a, mat_b, offs = args
        if mat_b.ndim != 3:
            raise ValueError("Original grouped B rank required")
        groups, k_size, n_size = mat_b.shape
        shape = (groups, n_size, k_size)
        if shape not in GROUPED_MM_SHAPES:
            raise ValueError("Original group_mm core descriptor required")
        expected = _group_mm_h800_config(dict(shape=list(shape)), torch.bfloat16)
        target = torch.device(self.device)
        if target.index is None:
            target = torch.device(target.type, torch_device_fn.current_device())
        for tensor, spec in zip(args, expected["inputs"]["args"]):
            if (
                list(tensor.shape) != spec["shape"]
                or str(tensor.dtype) != spec["dtype"]
                or list(tensor.stride()) != spec["stride"]
                or tensor.storage_offset() != 0
                or tensor.requires_grad
                or tensor.device != target
            ):
                raise ValueError(
                    "Original target shape/dtype/layout/offset/device required"
                )
        if len({t.untyped_storage().data_ptr() for t in args}) != 3:
            raise ValueError("Original separate A/B/offsets required")
        if offs.cpu().tolist() != expected["offsets"]:
            raise ValueError("Actual original cumulative group offsets required")
        expected_id = (
            "benchmark/test_grouped_mm.py::test_grouped_mm::core::bfloat16::"
            + str(GROUPED_MM_SHAPES.index(shape))
        )
        if case_id != expected_id:
            raise ValueError("Original ordered joint core identity required")
        baseline = base.Config.base_data
        if (
            hashlib.sha256(Path(baseline.source).read_bytes()).hexdigest()
            != _GROUP_MM_H800_SHA256
        ):
            raise ValueError("Exact reviewed native-composite H800 artifact required")
        data = baseline.data
        task = data["_collection"]["tasks"][self.op_name]
        env = data["_environments"][str(task["device"])]
        if (
            data[self.op_name]["native_api"] != "ops.group_mm"
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != _GROUP_MM_H800_SOURCE
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
        ):
            raise ValueError(
                "Completed clean native-composite H800 provenance required"
            )
        call = describe_call(args, kwargs)
        expected_call = dict(
            args=[
                {k: v for k, v in spec.items() if k != "storage_offset"}
                for spec in expected["inputs"]["args"]
            ],
            kwargs={},
        )
        record, key, reason = baseline.lookup(
            self.op_name, mat_a.dtype, call, target_device=self.device
        )
        if (
            record is None
            or reason is not None
            or record["config"] != expected
            or call != expected_call
            or not positive_number(record.get("T_us"))
            or not positive_number(record.get("latency_ms"))
            or abs(record["T_us"] / 1000 - record["latency_ms"]) > 1e-12
            or record.get("timing", {}).get("return_mode") != "median"
        ):
            raise ValueError(
                "Actual call must select exact original native-composite record"
            )
        return baseline, json.dumps(call, sort_keys=True), record["latency_ms"], key

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name in {"hygon", "metax"} and op is self.torch_op:
            raise ValueError("Selected backend requires explicit CPU-oracle candidate")
        return op, dispatch, overridden

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name in {"hygon", "metax"}:
            self._composite_checked = self._require_composite_static(
                *self.unpack_to_args_kwargs(input), case_id
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if flag_gems.vendor_name not in {"hygon", "metax"}:
            return super()._apply_external_baseline(metric, dtype, context)
        baseline, captured, latency, key = self._composite_checked
        if (
            context is None
            or context[0] is not baseline
            or base.Config.base_data is not baseline
            or json.dumps(context[1], sort_keys=True) != captured
            or hashlib.sha256(Path(baseline.source).read_bytes()).hexdigest()
            != _GROUP_MM_H800_SHA256
        ):
            raise ValueError(
                "Native-composite H800 source/call changed around candidate timer"
            )
        super()._apply_external_baseline(metric, dtype, context)
        if (
            metric.error_msg is not None
            or metric.latency_base is not None
            or metric.base_miss_reason is not None
            or metric.base_data_source != baseline.source
            or metric.base_shape_key != key
            or metric.base_latency_ms != latency
        ):
            raise ValueError("Native-composite H800 baseline consumption failed")


class GroupedMMBenchmark(_GroupMMH800Reference, OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = GROUPED_MM_SHAPES
        self.shape_desc = "groups, N, K"

    def get_tflops(self, op, *args, **kwargs):
        del op, kwargs
        mat_b, offs = args[1], args[2]
        groups, k_size, n_size = mat_b.shape
        boundaries = [0] + offs.cpu().tolist()
        return sum(
            (boundaries[idx + 1] - boundaries[idx]) * n_size * k_size * 2
            for idx in range(groups)
        )


def _case_fn(shape, dtype):
    del dtype
    groups, n_size, k_size = shape
    group_sizes = tuple(16 + (group_idx * 13) % 48 for group_idx in range(groups))
    yield base.BenchmarkCasePlan(
        shape={
            "mat_a": [sum(group_sizes), k_size],
            "mat_b": [groups, k_size, n_size],
            "offs": [groups],
        },
        params={"group_sizes": list(group_sizes)},
        builder_args=(groups, n_size, k_size, group_sizes),
    )


def _build_inputs_fn(plan, dtype, device):
    groups, n_size, k_size, group_sizes = plan.builder_args
    torch.manual_seed(0)
    mat_a = torch.rand((sum(group_sizes), k_size), dtype=dtype, device=device)
    mat_b = torch.rand((groups, k_size, n_size), dtype=dtype, device=device)
    offs = torch.tensor(
        [sum(group_sizes[: idx + 1]) for idx in range(groups)],
        dtype=torch.int32,
        device=device,
    )
    return mat_a, mat_b, offs


@pytest.mark.grouped_mm
@pytest.mark.group_mm
@pytest.mark.skip_native(
    vendors=["hygon", "metax"],
    reason="Selected backend uses exact native per-group mm composite H800 static records",
)
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon" and utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm():
    GroupedMMBenchmark(
        op_name="group_mm",
        torch_op=(
            _native_group_mm_composite
            if flag_gems.vendor_name in {"nvidia", "hygon", "metax"}
            else torch._grouped_mm
        ),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.bfloat16],
    ).run()
