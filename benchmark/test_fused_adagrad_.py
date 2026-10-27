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

import ast
import hashlib
import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import flag_gems
import pytest
import torch

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(256, 256), (1024, 256), (2048, 512), (65536,)]


SOURCE_COMMIT = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
EXECUTION_KIND = "native_cpu_with_device_transfers"
_ADAGRAD_HOST_COLLECTION_SOURCE = "aaf09163fd13be4b2daad1d2d6e024d38c073335"
_ADAGRAD_HOST_ARTIFACT_SHA256 = (
    "6fb36802c09121f6c5b5d4aac38428f724c90a4d154ef8920571e6d0194bd129"
)
_ADAGRAD_HOST_PROVENANCE_SHA256 = (
    "a7db7a292191fb103acb171baa2e283cf0642e1b20c62687287598233300364f"
)
_ADAGRAD_HOST_ARTIFACT_URL = "https://raw.githubusercontent.com/zhaojs12345/ops-baseline/60adf2d0603d5b0ba83858a7a09c58e182c2e9d2/op_pytorch_h800_fused_adagrad_host_current_core_20261028.json"
_ADAGRAD_HOST_PROVENANCE_URL = "https://raw.githubusercontent.com/zhaojs12345/ops-baseline/60adf2d0603d5b0ba83858a7a09c58e182c2e9d2/op_pytorch_h800_fused_adagrad_host_current_core_20261028.provenance.json"

_ADAGRAD_HOST_POLICY = dict(
    warmup_samples=1,
    minimum_samples=5,
    minimum_total_ms=100.0,
    maximum_samples=1000,
    cpu_threads=1,
    input_threads=2,
    statistic="median",
    clock="perf_counter_ns",
    fresh_state="full target clone before every single call; cloning/sync/hash untimed",
)


def _adagrad_tree_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, list):
        return [_adagrad_tree_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_adagrad_tree_clone(item) for item in value)
    if isinstance(value, dict):
        return {key: _adagrad_tree_clone(item) for key, item in value.items()}
    return value


def _adagrad_fingerprint(value):
    if isinstance(value, torch.Tensor):
        raw = (
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        return dict(
            type="tensor",
            device=str(value.device),
            dtype=str(value.dtype),
            shape=list(value.shape),
            stride=list(value.stride()),
            layout=str(value.layout),
            values_sha256=hashlib.sha256(raw).hexdigest(),
        )
    if isinstance(value, (list, tuple)):
        return dict(
            type=type(value).__name__,
            items=[_adagrad_fingerprint(item) for item in value],
        )
    if isinstance(value, dict):
        return dict(
            type="dict",
            items={key: _adagrad_fingerprint(item) for key, item in value.items()},
        )
    return dict(
        type=type(value).__name__,
        value=value.hex() if isinstance(value, float) else value,
    )


def _adagrad_host_samples(fn, args, kwargs, synchronize):
    policy = dict(_ADAGRAD_HOST_POLICY)
    previous_threads = torch.get_num_threads()
    template = _adagrad_fingerprint((args, kwargs))
    samples, warmups = [], []
    try:
        while (
            len(warmups) < policy["warmup_samples"]
            or len(samples) < policy["minimum_samples"]
            or sum(item["elapsed_ns"] for item in samples) / 1e6
            < policy["minimum_total_ms"]
        ):
            if len(samples) >= policy["maximum_samples"]:
                raise RuntimeError("Adagrad host sample budget not reached")
            torch.set_num_threads(policy["input_threads"])
            fresh_args, fresh_kwargs = _adagrad_tree_clone((args, kwargs))
            synchronize()
            before = _adagrad_fingerprint((fresh_args, fresh_kwargs))
            if before != template:
                raise RuntimeError(
                    "Adagrad sample did not start from the complete original state"
                )
            lists = fresh_args[:4]
            frozen = [
                (
                    group,
                    tuple(
                        (
                            tensor,
                            tensor.data_ptr(),
                            tensor.device,
                            tensor.dtype,
                            tuple(tensor.shape),
                            tuple(tensor.stride()),
                            tensor.storage_offset(),
                        )
                        for tensor in group
                    ),
                )
                for group in lists
            ]
            torch.set_num_threads(policy["cpu_threads"])
            synchronize()
            started = time.perf_counter_ns()
            result = fn(*fresh_args, **fresh_kwargs)
            synchronize()
            elapsed = time.perf_counter_ns() - started
            if result is not None or elapsed <= 0:
                raise RuntimeError("Adagrad sample must execute one void-return call")
            for group, originals in frozen:
                if len(group) != len(originals):
                    raise RuntimeError("Adagrad candidate replaced an input list")
                for tensor, (
                    original,
                    pointer,
                    device,
                    dtype,
                    shape,
                    stride,
                    offset,
                ) in zip(group, originals):
                    if (
                        tensor is not original
                        or tensor.data_ptr() != pointer
                        or (
                            tensor.device,
                            tensor.dtype,
                            tuple(tensor.shape),
                            tuple(tensor.stride()),
                            tensor.storage_offset(),
                        )
                        != (device, dtype, shape, stride, offset)
                    ):
                        raise RuntimeError(
                            "Adagrad candidate changed original target ABI/alias/layout"
                        )
            after = _adagrad_fingerprint((fresh_args, fresh_kwargs))
            # Only parameter and state_sum values may change.
            for index in (1, 3):
                if (
                    before["items"][0]["items"][index]
                    != after["items"][0]["items"][index]
                ):
                    raise RuntimeError(
                        "Adagrad candidate mutated readonly gradients/steps"
                    )
            if before["items"][1] != after["items"][1]:
                raise RuntimeError("Adagrad candidate mutated options")
            row = dict(
                elapsed_ns=elapsed,
                input_state_sha256=hashlib.sha256(
                    json.dumps(before, sort_keys=True).encode()
                ).hexdigest(),
                single_call=True,
                original_target_abi_preserved=True,
                readonly_preserved=True,
            )
            (warmups if len(warmups) < policy["warmup_samples"] else samples).append(
                row
            )
        if _adagrad_fingerprint((args, kwargs)) != template:
            raise RuntimeError("Adagrad timing mutated the original template")
    finally:
        torch.set_num_threads(previous_threads)
    values = sorted(row["elapsed_ns"] for row in samples)
    count = len(values)
    median_ns = (values[(count - 1) // 2] + values[count // 2]) / 2
    return dict(
        latency_ms=median_ns / 1e6,
        measurement_method="synchronized_host_wall_single_call_fresh_state",
        policy=policy,
        samples=samples,
        warmups=warmups,
        original_template_unchanged=True,
        actual_input_fingerprint=template,
    )


def _adagrad_reference_config(case):
    shape = list(case["shape"])
    stride = []
    product = 1
    for size in reversed(shape):
        stride.insert(0, product)
        product *= size
    tensor = dict(
        dtype="torch.float32",
        shape=shape,
        stride=stride,
        layout="torch.strided",
        requires_grad=False,
        device="target",
        storage_offset=0,
    )
    return dict(
        operator="_fused_adagrad_",
        source_commit=SOURCE_COMMIT,
        benchmark="benchmark/test_fused_adagrad_.py",
        scope="core",
        descriptor=shape,
        mode=case["mode"],
        dtype="torch.float32",
        native_api="torch._fused_adagrad_",
        execution_kind=EXECUTION_KIND,
        inputs=dict(
            params=[tensor],
            grads=[tensor],
            state_sums=[tensor],
            state_steps=[
                dict(
                    dtype="torch.float32",
                    shape=[1],
                    stride=[1],
                    layout="torch.strided",
                    requires_grad=False,
                    device="target",
                    storage_offset=0,
                    scalar_value=3.0,
                )
            ],
        ),
        options=dict(
            lr=0.01,
            lr_decay=case["lr_decay"],
            weight_decay=case["weight_decay"],
            eps=1e-10,
            maximize=case["maximize"],
            grad_scale=None,
            found_inf=None,
        ),
        input_recipe=dict(
            seed=0,
            params="target randn",
            grads="target randn",
            state_sums="target rand + 0.25",
            state_steps="target float32 [3.0]",
            cross_chip_values="same recipe; GPU random values not claimed equal",
        ),
        effects=dict(
            mutates=["params", "state_sums"],
            readonly=["grads", "state_steps", "grad_scale", "found_inf"],
            returns=None,
        ),
        measurement_policy=_ADAGRAD_HOST_POLICY,
    )


@dataclass
class _AdagradHostMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None
    host_measurement: dict | None = None


class FusedAdagradBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "optimizer parameter shape"

    @cached_property
    def _adagrad_host_reference(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        proof_path = Path(os.environ["FLAGGEMS_NV_BASELINE_PROVENANCE"]).resolve(
            strict=True
        )
        payload, proof_payload = path.read_bytes(), proof_path.read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != _ADAGRAD_HOST_ARTIFACT_SHA256
            or hashlib.sha256(proof_payload).hexdigest()
            != _ADAGRAD_HOST_PROVENANCE_SHA256
        ):
            raise ValueError(
                "Adagrad requires its exact reviewed CPU-native H800 host baseline"
            )

        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError("Duplicate host baseline key")
                out[key] = value
            return out

        def nonfinite(value):
            raise ValueError("Nonfinite host baseline value: " + value)

        data = json.loads(payload, object_pairs_hook=unique, parse_constant=nonfinite)
        proof = json.loads(
            proof_payload, object_pairs_hook=unique, parse_constant=nonfinite
        )
        environment, execution = data["_environment"], data["_execution"]
        if (
            data["schema"] != "adagrad.host-baseline/v1"
            or data["operator"] != "_fused_adagrad_"
            or data["execution_kind"] != EXECUTION_KIND
            or data["native_api"] != "torch._fused_adagrad_"
            or data["dispatch"] != {"CPU": True, "CUDA": False}
        ):
            raise ValueError("Wrong Adagrad native API or host execution kind")
        if (
            execution["collection_source"] != _ADAGRAD_HOST_COLLECTION_SOURCE
            or environment["source"]["git_commit"] != _ADAGRAD_HOST_COLLECTION_SOURCE
            or environment["source"]["git_dirty"] is not False
            or execution["source_before_after_exact"] is not True
            or execution["input_source"] != SOURCE_COMMIT
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["hardware"]["gpu"]["uuid"]
            != "f76856dc-d02f-95a0-dc05-054190d1ea7d"
            or not environment["hardware"]["cpu_model"]
        ):
            raise ValueError("Incomplete actual H800/CPU/source provenance")
        if (
            environment["run"]["execution_kind"] != EXECUTION_KIND
            or environment["run"]["ncu_enabled"] is not False
            or environment["run"]["policy"] != _ADAGRAD_HOST_POLICY
        ):
            raise ValueError("Adagrad host measurement policy differs")
        if (
            proof["status"] != "PASS_VERIFIED_ADAGRAD_CPU_NATIVE_H800_TRANSFERS"
            or proof["artifact_sha256"] != _ADAGRAD_HOST_ARTIFACT_SHA256
            or proof["collection_source"] != _ADAGRAD_HOST_COLLECTION_SOURCE
            or proof["native_records"] != 8
            or proof["remeasured"] is not False
        ):
            raise ValueError(
                "Adagrad host provenance does not bind original eight records"
            )
        identities = {}
        test_path = Path(__file__).resolve().parents[1] / "tests/test_fused_adagrad_.py"
        text = test_path.read_text()
        functions = {
            node.name: node
            for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef)
        }
        for name in [
            "_adagrad_dense",
            "_adagrad_validate",
            "_adagrad_cpu_copy",
            "_adagrad_cpu_native",
            "_adagrad_cpu_apply",
        ]:
            identities[name] = hashlib.sha256(
                (ast.get_source_segment(text, functions[name]) + "\n").encode()
            ).hexdigest()
        for fn in [_adagrad_tree_clone, _adagrad_fingerprint, _adagrad_host_samples]:
            identities[fn.__name__] = hashlib.sha256(
                inspect.getsource(fn).encode()
            ).hexdigest()
        if execution["source_helper_identity"] != identities:
            raise ValueError("Adagrad CPU wrapper or fresh-state timer source differs")
        rows = data["records"]
        expected_ids = {
            f"benchmark/test_fused_adagrad_.py::test_fused_adagrad_::core::float32::{index}"
            for index in range(8)
        }
        if len(rows) != 8 or {row["case_id"] for row in rows} != expected_ids:
            raise ValueError("Missing/duplicate original Adagrad core record")
        return rows, dict(
            kind=EXECUTION_KIND,
            artifact_sha256=_ADAGRAD_HOST_ARTIFACT_SHA256,
            provenance_sha256=_ADAGRAD_HOST_PROVENANCE_SHA256,
            artifact_url=_ADAGRAD_HOST_ARTIFACT_URL,
            provenance_url=_ADAGRAD_HOST_PROVENANCE_URL,
            collection_source=_ADAGRAD_HOST_COLLECTION_SOURCE,
            environment=environment,
            ratio="H800 CPU-native+transfer host ms / actual target candidate host ms; no scaling or GPU-kernel claim",
        )

    def _import_adagrad_host(self, args, kwargs):
        case_id = getattr(self, "_adagrad_case_id", None)
        if getattr(self, "_adagrad_reference_traversal", False):
            active = [
                row
                for row in base.Config.reference_records
                if row.get("operator") == self.op_name
                and row.get("status") == "FAILED"
                and row.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise ValueError("Cannot identify active Adagrad reference case")
            case_id = active[0]["case_id"]
        cases = [case for case in self._collect_cases() if case.case_id == case_id]
        if len(cases) != 1 or cases[0].dtype != torch.float32:
            raise ValueError("Missing original Adagrad case identity")
        case = cases[0]
        current = dict(shape=tuple(case.shape["param"]), **case.params)
        config = _adagrad_reference_config(current)
        if len(args) != 4 or any(
            type(group) is not list or len(group) != 1 for group in args
        ):
            raise ValueError("Expected original four one-element Adagrad lists")
        target = torch.empty(0, device=self.device).device
        shape = tuple(current["shape"])
        stride = tuple(config["inputs"]["params"][0]["stride"])
        pointers = []
        for index, group in enumerate(args):
            tensor = group[0]
            expected_shape, expected_stride = (
                ((1,), (1,)) if index == 3 else (shape, stride)
            )
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device != target
                or tensor.dtype != torch.float32
                or tensor.layout != torch.strided
                or tuple(tensor.shape) != expected_shape
                or tuple(tensor.stride()) != expected_stride
                or tensor.storage_offset() != 0
                or tensor.requires_grad
            ):
                raise ValueError("Actual Adagrad input target ABI differs")
            pointers.append(tensor.data_ptr())
        if len(set(pointers)) != 4 or float(args[3][0].cpu().item()) != 3.0:
            raise ValueError("Adagrad original alias/step recipe differs")
        options = config["options"]
        if set(kwargs) != set(options) or any(
            type(kwargs[key]) is not type(value) or kwargs[key] != value
            for key, value in options.items()
        ):
            raise ValueError("Adagrad original typed scalar/options recipe differs")
        records, source = self._adagrad_host_reference
        matches = [
            row
            for row in records
            if row["case_id"] == case_id and row["config"] == config
        ]
        if len(matches) != 1:
            raise ValueError("Missing exact original Adagrad host record")
        row = matches[0]
        milliseconds, microseconds = row["latency_ms"], row["T_us"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in [milliseconds, microseconds]
        ) or not math.isclose(
            milliseconds, microseconds / 1000, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("Invalid Adagrad host latency units")
        if (
            row["host_measurement"]["policy"] != _ADAGRAD_HOST_POLICY
            or row["host_measurement"]["measurement_method"]
            != "synchronized_host_wall_single_call_fresh_state"
        ):
            raise ValueError("Wrong Adagrad host sampling method")
        self._adagrad_imported_source = dict(
            source,
            config=config,
            case_id=case_id,
            source_input_fingerprint=row["host_measurement"][
                "actual_input_fingerprint"
            ],
            input_identity="same exact seed/recipe/ABI; cross-chip random payload equality is not claimed",
        )
        if getattr(self, "_adagrad_reference_traversal", False):
            active[0]["baseline_source"] = self._adagrad_imported_source
        return milliseconds

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name not in ("enflame", "nvidia"):
            return super().get_latency(op, *args, **kwargs)
        reference = op is self.torch_op and (
            getattr(self, "_adagrad_native_pending", False)
            or (
                getattr(self, "_adagrad_reference_traversal", False)
                and base.Config.reference_only
            )
        )
        if reference:
            self._adagrad_native_pending = False
            if self.is_backward or base.Config.skip_native:
                raise ValueError(
                    "Adagrad host reference is forward-only with native reference enabled"
                )
            return self._import_adagrad_host(args, kwargs)
        self._adagrad_candidate_measurement = _adagrad_host_samples(
            op, args, kwargs, base.torch_device_fn.synchronize
        )
        return self._adagrad_candidate_measurement["latency_ms"]

    def _measure_input(self, input, case_id=None):
        self._adagrad_imported_source = None
        self._adagrad_candidate_measurement = None
        self._adagrad_case_id = case_id
        self._adagrad_native_pending = (
            flag_gems.vendor_name in ("enflame", "nvidia")
            and "latency_base" in self.to_bench_metrics
            and not base.Config.skip_native
        )
        try:
            metric = super()._measure_input(input, case_id)
            if (
                self._adagrad_imported_source is not None
                or self._adagrad_candidate_measurement is not None
            ):
                return _AdagradHostMetrics(
                    **asdict(metric),
                    baseline_source=self._adagrad_imported_source,
                    host_measurement=self._adagrad_candidate_measurement,
                )
            return metric
        finally:
            self._adagrad_native_pending = False
            self._adagrad_case_id = None

    def _run_reference_cases(self, case_ids):
        first = len(base.Config.reference_records)
        self._adagrad_reference_traversal = True
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            self._adagrad_reference_traversal = False
            self._adagrad_native_pending = False
            for row in base.Config.reference_records[first:]:
                if row.get("status") == "PASSED" and row.get("baseline_source"):
                    row["stage"] = "external_host_baseline"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for mode, lr_decay, weight_decay, maximize in (
        ("plain", 0.0, 0.0, False),
        ("options", 0.2, 0.01, True),
    ):
        yield base.BenchmarkCasePlan(
            shape={"param": list(dims)},
            params={
                "mode": mode,
                "lr_decay": lr_decay,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
            builder_args=(dims, lr_decay, weight_decay, maximize),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, lr_decay, weight_decay, maximize = plan.builder_args
    torch.manual_seed(0)
    params = [torch.randn(shape, dtype=dtype, device=device)]
    grads = [torch.randn(shape, dtype=dtype, device=device)]
    state_sums = [torch.rand(shape, dtype=dtype, device=device) + 0.25]
    state_steps = [torch.tensor([3.0], dtype=torch.float32, device=device)]
    kwargs = {
        "lr": 0.01,
        "lr_decay": lr_decay,
        "weight_decay": weight_decay,
        "eps": 1e-10,
        "maximize": maximize,
        "grad_scale": None,
        "found_inf": None,
    }
    return params, grads, state_sums, state_steps, kwargs


@pytest.mark.fused_adagrad_
@pytest.mark.underscore_fused_adagrad_
def test_fused_adagrad_():
    FusedAdagradBenchmark(
        op_name="_fused_adagrad_",
        torch_op=torch._fused_adagrad_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
        fresh_inputs=True,
    ).run()
