# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import math
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config

CORE_SHAPES = [
    (1, 32, False),
    (8, 64, True),
    (32, 128, False),
    (64, 256, True),
    (128, 512, False),
    (256, 1024, True),
]
COMPREHENSIVE_SHAPES = [
    (512, 1024, False),
    (512, 2048, True),
    (1024, 2048, True),
]
DTYPES = list(consts.FLOAT_DTYPES)
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)


class _H800GRUReference:
    # Exact unmodified official ops-baseline output; all three native overloads.
    def _require_h800(self):
        data = Config.base_data
        if data is None:
            raise RuntimeError(
                "Ascend GRU timing requires --base-data with the collected native H800 artifact"
            )
        payload = Path(data.source).read_bytes()
        if (
            hashlib.sha256(payload).hexdigest()
            != "2e42eeddb4bf7fcabdb4c695aa531c10c230b51a630625def6669f99bf694adb"
        ):
            raise ValueError("Unexpected GRU native H800 baseline artifact")
        env = data.data["_environment"]
        if (
            env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != "e3776c2c1d4431da168f18577c690d7f807451e7"
            or env["source"]["git_dirty"]
            or env["run"]["repo"] != "pytorch"
            or set(env["run"]["selected_ops"])
            != {
                "_thnn_fused_gru_cell",
                "_thnn_fused_gru_cell_out",
                "_thnn_fused_gru_cell_backward_out",
            }
        ):
            raise ValueError("GRU baseline is not the verified native H800 collection")

    def run(self, *args, **kwargs):
        if flag_gems.vendor_name == "ascend" and not (
            Config.query
            or Config.list_cases
            or Config.preflight_only
            or Config.profile_only
            or Config.reference_only
        ):
            self._require_h800()
            if not Config.skip_native:
                raise RuntimeError(
                    "Ascend GRU must not time its incomplete local native reference"
                )
        return super().run(*args, **kwargs)

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "ascend" and not overridden and op is self.torch_op:
            raise RuntimeError(
                "Ascend GRU requires an explicit candidate; native workspace is unavailable"
            )
        return op, dispatch, overridden

    def _apply_external_baseline(self, metric, dtype, context):
        super()._apply_external_baseline(metric, dtype, context)
        if flag_gems.vendor_name == "ascend":
            if (
                metric.latency_base is not None
                or metric.base_miss_reason is not None
                or not isinstance(metric.base_latency_ms, (int, float))
                or not math.isfinite(metric.base_latency_ms)
                or metric.base_latency_ms <= 0
            ):
                raise RuntimeError(
                    "GRU native H800 baseline did not exactly match this case"
                )


class FusedGRUCellBenchmark(_H800GRUReference, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += list(COMPREHENSIVE_SHAPES)

    def get_case_iter(self, cur_dtype):
        for ordinal, (batch_size, hidden_size, with_bias) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input_gates": [batch_size, 3 * hidden_size]},
                    params={"with_bias": with_bias},
                    builder_args=(batch_size, hidden_size, with_bias),
                ),
            )

    def build_inputs(self, case):
        batch_size, hidden_size, with_bias = case.builder_args[0].builder_args
        cur_dtype = case.dtype
        input_gates = torch.randn(
            batch_size,
            3 * hidden_size,
            dtype=cur_dtype,
            device=self.device,
        )
        hidden_gates = torch.randn_like(input_gates)
        hx = torch.randn(
            batch_size,
            hidden_size,
            dtype=cur_dtype,
            device=self.device,
        )
        if with_bias:
            input_bias = torch.randn(
                3 * hidden_size, dtype=cur_dtype, device=self.device
            )
            hidden_bias = torch.randn_like(input_bias)
        else:
            input_bias = None
            hidden_bias = None
        return input_gates, hidden_gates, hx, input_bias, hidden_bias


class FusedGRUCellOutBenchmark(_H800GRUReference, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(batch, hidden) for batch, hidden, _ in CORE_SHAPES]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [
                (batch, hidden) for batch, hidden, _ in COMPREHENSIVE_SHAPES
            ]

    def get_case_iter(self, cur_dtype):
        for ordinal, (batch_size, hidden_size) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input_gates": [batch_size, 3 * hidden_size]},
                    params={"with_bias": True},
                    builder_args=(batch_size, hidden_size),
                ),
            )

    def build_inputs(self, case):
        batch_size, hidden_size = case.builder_args[0].builder_args
        cur_dtype = case.dtype
        input_gates = torch.randn(
            batch_size, 3 * hidden_size, dtype=cur_dtype, device=self.device
        )
        hidden_gates = torch.randn_like(input_gates)
        hx = torch.randn(batch_size, hidden_size, dtype=cur_dtype, device=self.device)
        input_bias = torch.randn(3 * hidden_size, dtype=cur_dtype, device=self.device)
        hidden_bias = torch.randn_like(input_bias)
        out0 = torch.empty_like(hx)
        out1 = torch.empty(
            batch_size, 5 * hidden_size, dtype=cur_dtype, device=self.device
        )
        return (
            input_gates,
            hidden_gates,
            hx,
            input_bias,
            hidden_bias,
            {"out0": out0, "out1": out1},
        )


@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native GRU workspace unavailable; use the verified H800 --base-data comparison",
)
@pytest.mark.underscore_thnn_fused_gru_cell
def test_thnn_fused_gru_cell():
    bench = FusedGRUCellBenchmark(
        op_name="_thnn_fused_gru_cell",
        torch_op=torch.ops.aten._thnn_fused_gru_cell.default,
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native GRU workspace unavailable; use the verified H800 --base-data comparison",
)
@pytest.mark.underscore_thnn_fused_gru_cell_out
def test_thnn_fused_gru_cell_out():
    bench = FusedGRUCellOutBenchmark(
        op_name="_thnn_fused_gru_cell_out",
        torch_op=torch.ops.aten._thnn_fused_gru_cell.out,
        dtypes=DTYPES,
    )
    bench.run()
