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

import pytest
import torch

import flag_gems

from . import base, consts

# These skips describe the currently validated runtime stacks, not every device
# from these vendors. Revisit them when native support is added to those stacks.
_NATIVE_UNSUPPORTED = {
    "kunlunxin": "current XPU runtime routes native FP8/FP16 grouped MM to an unimplemented CPU fallback",
    "ascend": "CANN 9.0 has no native aten::_scaled_grouped_mm PrivateUse1 kernel",
    "hygon": "current DTK native grouped MM rejects the tested architecture",
    "metax": "MACA 3.8.1 reports grouped mm is not supported on your system",
    "thead": "current CI native grouped MM rejects the tested architecture",
    "tsingmicro": "current TXDA native grouped MM falls back to unsupported CPU",
    "iluvatar": "BI-V150/CoreX 4.4 does not support native grouped MM",
    "sunrise": "current CI reports native torch._scaled_grouped_mm unavailable",
    "enflame": "current CI reports native torch._scaled_grouped_mm unavailable",
}


class ScaledGroupedMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self):
        super().__init__(
            "scaled_grouped_mm", torch._scaled_grouped_mm, dtypes=[torch.float8_e4m3fn]
        )
        self.set_gems(flag_gems.scaled_grouped_mm)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 64, 128, 128),
            (8, 128, 256, 256),
            (16, 256, 512, 512),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))
        self.shape_desc = "groups, M_per_group, N, K"

    def set_more_shapes(self):
        return [
            (16, 512, 1024, 1024),
            (32, 256, 2048, 1024),
        ]

    def get_input_iter(self, dtype):
        for groups, m_per_group, N, K in self.shapes:
            sizes = torch.arange(
                m_per_group,
                m_per_group + groups,
                dtype=torch.int32,
                device="cpu",
            )
            offs = torch.cumsum(sizes, dim=0).to(torch.int32)
            M = int(offs[-1].item())

            mat_a = torch.randn((M, K), dtype=torch.float32, device="cpu")
            mat_b = torch.randn((groups, N, K), dtype=torch.float32, device="cpu")
            mat_a = (mat_a * 0.25).to(dtype).to(flag_gems.device)
            mat_b = (mat_b * 0.25).to(dtype).to(flag_gems.device).transpose(-1, -2)
            out_dtype = torch.bfloat16

            scale_a = torch.linspace(0.75, 1.25, M, device="cpu").to(flag_gems.device)
            scale_b = (
                torch.linspace(1.25, 0.75, groups * N, device="cpu")
                .to(flag_gems.device)
                .reshape(groups, N)
            )
            yield mat_a, mat_b, scale_a, scale_b, {
                "offs": offs.to(flag_gems.device),
                "bias": None,
                "out_dtype": out_dtype,
                "use_fast_accum": False,
            }

    def get_tflops(self, op, *args, **kwargs):
        mat_b = args[1]
        offs = kwargs["offs"]
        groups, K, N = mat_b.shape
        sizes = torch.diff(
            offs, prepend=torch.zeros(1, device=offs.device, dtype=offs.dtype)
        )
        total_flops = 0
        for group_idx in range(groups):
            total_flops += int(sizes[group_idx].item()) * N * K * 2
        return total_flops


@pytest.mark.scaled_grouped_mm
@pytest.mark.skipif(
    flag_gems.vendor_name in _NATIVE_UNSUPPORTED,
    reason=_NATIVE_UNSUPPORTED.get(flag_gems.vendor_name, "native baseline available"),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "nvidia"
    and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] not in (9, 10)
    ),
    reason="NVIDIA native scaled grouped MM requires SM9x or SM10x",
)
def test_scaled_grouped_mm_benchmark():
    bench = ScaledGroupedMMBenchmark()
    bench.run()
