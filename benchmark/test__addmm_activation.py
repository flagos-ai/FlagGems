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

# The native GCU APIs dispatch these inputs through CPU ATen. Import the
# exact H800 reference once per reference phase; the candidate still runs locally.
_H800_HEAD = "8e9ebf8b4c66a362f7bbc6523a4c7cff57b139d1"
_H800_COLLECTION = "7ef733945b962d749c759ec2e6021d64d78cbe3b"
_H800_FILE = "op_pytorch_h800_b03_ac9_exact_latency.json"
_H800_CASES = "op_pytorch_h800_b03_ac9_exact_cases.json"
_H800_SHA = "ef34f907b602d119f5921a69d215706115c4fcc0c3bce795741f27868de2d08f"
_H800_CASES_SHA = "e9573708a3b057e12fdde3b97cba645fcab9fb50daffefb8de08d04234d95d2a"


@dataclass
class _B03Metrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _B03H800ReferenceMixin:
    @cached_property
    def _h800_reference(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        index_path = Path(
            os.environ.get(
                "FLAGGEMS_NV_BASELINE_PROVENANCE", str(path.with_name(_H800_CASES))
            )
        ).resolve(strict=True)

        def load(path, digest):
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("B03 H800 snapshot hash mismatch")

            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate B03 baseline key: " + key)
                    result[key] = value
                return result

            def reject(value):
                raise ValueError("Non-finite B03 baseline value: " + value)

            return json.loads(payload, object_pairs_hook=unique, parse_constant=reject)

        data, index = load(path, _H800_SHA), load(index_path, _H800_CASES_SHA)
        environment, packaging = data["_environment"], data["_packaging"]
        if (
            environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _H800_COLLECTION
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["repo"] != "pytorch"
            or packaging["collection_source_commit"] != _H800_COLLECTION
            or packaging["case_index_sha256"] != _H800_CASES_SHA
            or index["schema"] != "b03-exact-native-case-index/v1"
        ):
            raise ValueError("B03 requires its clean exact native H800 collection")
        cases = {}
        for case in index["cases"]:
            key = case["case_id"]
            if key in cases:
                raise ValueError("Duplicate B03 case identity")
            row = data[case["module"]]["shapes"][case["record_shape_key"]][
                case["dtype"]
            ]
            latency = row["T_us"]
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency <= 0
                or not math.isclose(
                    row["latency_ms"], latency / 1000, rel_tol=1e-9, abs_tol=1e-12
                )
                or row["config"]["binding"] != case["binding"]
                or row["config"]["dtype"] != case["dtype"]
            ):
                raise ValueError("Invalid exact B03 latency record")
            cases[key] = (case, row)
        counts = {
            name: sum(c[0]["op_name"] == name for c in cases.values())
            for name in ("_addmm_activation", "_addmm_activation_out", "addbmm")
        }
        if len(cases) != 180 or list(counts.values()) != [75, 75, 30]:
            raise ValueError("B03 snapshot must preserve all 75/75/30 original cases")
        return cases, environment, packaging

    def _external_case(self):
        if getattr(self, "_reference_traversal", False):
            active = [
                row
                for row in base.Config.reference_records
                if row.get("operator") == self.op_name
                and row.get("status") == "FAILED"
                and row.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError("Cannot identify active B03 reference case")
            return active[0]["case_id"], active[0]
        return self._active_case_id, None

    def get_latency(self, op, *args, **kwargs):
        external = (
            flag_gems.vendor_name == "enflame"
            and op is self.torch_op
            and (
                getattr(self, "_native_pending", False)
                or (
                    getattr(self, "_reference_traversal", False)
                    and base.Config.reference_only
                )
            )
        )
        if not external:
            return super().get_latency(op, *args, **kwargs)
        self._native_pending = False  # Consume before loading, including failures.
        if self.is_backward or base.Config.skip_native or len(args) != 3:
            raise ValueError("External B03 reference requires the original forward ABI")
        case_id, active = self._external_case()
        cases, environment, packaging = self._h800_reference
        indexed, row = cases[case_id]
        local = [case for case in self._collect_cases() if case.case_id == case_id]
        if (
            len(local) != 1
            or indexed["op_name"] != self.op_name
            or local[0].shape != indexed["shape"]
            or local[0].params != indexed["params"]
            or str(local[0].dtype) != indexed["dtype"]
        ):
            raise ValueError("B03 original joint case identity differs")
        config = row["config"]
        legacy = config["source_file"] == "benchmark/test__addmm_activation.py"
        expected_source = (
            "benchmark/test__addmm_activation.py"
            if legacy
            else {
                "_addmm_activation": "benchmark/test_addmm_activation.py",
                "_addmm_activation_out": "benchmark/test_addmm_activation_out.py",
                "addbmm": "benchmark/test_addbmm.py",
            }[self.op_name]
        )
        native = (
            torch.addbmm
            if self.op_name == "addbmm"
            else (
                torch.ops.aten._addmm_activation.out
                if self.op_name == "_addmm_activation_out" and not legacy
                else torch._addmm_activation
            )
        )
        names = (
            ("self", "batch1", "batch2")
            if self.op_name == "addbmm"
            else ("self", "mat1", "mat2")
        )
        tensors = dict(zip(names, args))
        expected_kwargs = (
            {}
            if legacy or self.op_name == "addbmm"
            else {"use_gelu": config["use_gelu"]}
        )
        if self.op_name == "_addmm_activation_out":
            tensors["out"] = kwargs.get("out")
            expected_kwargs["out"] = tensors["out"]
        if (
            op is not native
            or config["source_file"] != expected_source
            or set(kwargs) != set(expected_kwargs)
            or any(
                kwargs[key] != value
                for key, value in expected_kwargs.items()
                if key != "out"
            )
            or config["alpha"] != 1
            or config["beta"] != 1
            or set(tensors) != set(config["tensor_shapes"])
        ):
            raise ValueError(
                "B03 native callable/scalars differ from the original recipe"
            )
        for name, value in tensors.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "gcu"
                or value.device != args[0].device
                or str(value.dtype) != indexed["dtype"]
                or value.requires_grad
                or list(value.shape) != config["tensor_shapes"][name]
                or list(value.stride()) != config["tensor_strides"][name]
            ):
                raise ValueError(
                    "B03 target tensor ABI differs from the original recipe"
                )
        prefix = (
            "https://raw.githubusercontent.com/zhaojs12345/ops-baseline/"
            + _H800_HEAD
            + "/"
        )
        self._imported_reference = {
            "kind": "external_nvidia",
            "case_id": case_id,
            "module": indexed["module"],
            "config": config,
            "environment": environment,
            "packaging": packaging,
            "artifact_url": prefix + _H800_FILE,
            "case_index_url": prefix + _H800_CASES,
            "artifact_sha256": _H800_SHA,
            "case_index_sha256": _H800_CASES_SHA,
            "collection_source_commit": _H800_COLLECTION,
            "ratio": "raw H800 latency / local candidate latency; no peak normalization",
            "input_identity": "original joint recipe; legacy ambient RNG is not historical/cross-chip tensor identity",
        }
        if active is not None:
            active["baseline_source"] = self._imported_reference
        return row["T_us"] / 1000

    def _measure_input(self, input, case_id=None):
        self._active_case_id, self._imported_reference = case_id, None
        self._native_pending = (
            flag_gems.vendor_name == "enflame"
            and "latency_base" in self.to_bench_metrics
            and not base.Config.skip_native
        )
        try:
            metric = super()._measure_input(input, case_id)
            if self._imported_reference is not None:
                return _B03Metrics(
                    **asdict(metric), baseline_source=self._imported_reference
                )
            return metric
        finally:
            self._native_pending = False
            self._active_case_id = None

    def _run_reference_cases(self, case_ids):
        first = len(base.Config.reference_records)
        self._reference_traversal = True
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            self._reference_traversal = False
            self._native_pending = False
            for row in base.Config.reference_records[first:]:
                if row.get("baseline_source") and row.get("status") == "PASSED":
                    row["stage"] = "external_baseline"


class _B03LegacyBenchmark(_B03H800ReferenceMixin, base.BlasBenchmark):
    pass


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2,


@pytest.mark.addmm_activation
@pytest.mark.underscore_addmm_activation
def test_addmm_activation(monkeypatch):
    bench = _B03LegacyBenchmark(
        op_name="_addmm_activation",
        input_fn=_input_fn,
        torch_op=torch._addmm_activation,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    out = torch.empty([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2, {"out": out}


@pytest.mark.addmm_activation_out
@pytest.mark.underscore_addmm_activation_out
def test_addmm_activation_out(monkeypatch):
    bench = _B03LegacyBenchmark(
        op_name="_addmm_activation_out",
        input_fn=_input_fn_out,
        torch_op=torch._addmm_activation,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
