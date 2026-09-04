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

import gc

import pytest
import torch

import flag_gems

from . import base
from .conftest import Config, emit_record_logger, update_result
from .consts import BenchmarkMetrics, BenchmarkResult

# Benchmark shapes: (M, N, K, qGroupSize)
# M=activation rows, N=output channels, K=feature dim, qGroupSize=quantization group size
WEIGHT_INT4PACK_MM_SHAPES = [
    (16, 32, 128, 64),
    (32, 64, 256, 64),
    (64, 128, 256, 64),
    (64, 128, 512, 128),
    (128, 256, 512, 128),
    (128, 256, 1024, 256),
    (256, 512, 1024, 256),
    (512, 1024, 2048, 256),
]


def _torch_reference_int4pack_mm(A, mat2_packed, qGroupSize, qScaleAndZeros):
    """Eager-PyTorch baseline using the same byte-pair int4 packing.

    The native torch._weight_int4pack_mm expects the Marlin tiled format, which
    is incompatible with FlagGems' byte-pair packing (matching the sibling op
    _convert_weight_to_int4pack). So the baseline dequantizes with vectorized
    tensor ops and runs a dense matmul — a fair "naive PyTorch" reference for
    measuring the fused Triton kernel's speedup.
    """
    N, K_half = mat2_packed.shape
    K = K_half * 2
    low = (mat2_packed & 0xF).to(torch.int32)
    high = ((mat2_packed >> 4) & 0xF).to(torch.int32)
    q = torch.empty((N, K), dtype=torch.int32, device=mat2_packed.device)
    q[:, 0::2] = low
    q[:, 1::2] = high
    # (num_groups, N) -> (N, K) per-element scale/zero
    scales = qScaleAndZeros[:, :, 0].t().repeat_interleave(qGroupSize, dim=1)
    zeros = qScaleAndZeros[:, :, 1].t().repeat_interleave(qGroupSize, dim=1)
    w_dequant = (q.to(A.dtype) - zeros) * scales
    return A @ w_dequant.t()


def _weight_int4pack_mm_input_fn(shape, dtype, device):
    """Yield input tuples for _weight_int4pack_mm benchmark."""
    M, N, K, qGroupSize = shape
    # Create activation tensor
    A = torch.randn((M, K), dtype=dtype, device=device)
    # Create int4 weights (values 0..15)
    weight_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    # Pack weights into byte-pair format
    packed = torch.empty((N, K // 2), dtype=torch.uint8, device=device)
    for n in range(N):
        for k_half in range(K // 2):
            even = weight_int4[n, 2 * k_half].item() & 0xF
            odd = weight_int4[n, 2 * k_half + 1].item() & 0xF
            packed[n, k_half] = (odd << 4) | even
    # Create scales and zeros
    num_groups = K // qGroupSize
    scales = torch.rand((num_groups, N), dtype=dtype, device=device) * 1.5 + 0.5
    zeros = torch.randint(4, 11, (num_groups, N), dtype=dtype, device=device)
    qScaleAndZeros = torch.stack([scales, zeros], dim=-1)
    yield A, packed, qGroupSize, qScaleAndZeros


class WeightInt4PackMmBenchmark(base.Benchmark):
    """Benchmark for _weight_int4pack_mm operator.

    The native torch._weight_int4pack_mm uses the Marlin tiled weight format,
    incompatible with FlagGems' byte-pair int4 packing. The baseline is a naive
    eager-PyTorch dequant + dense matmul over the same packing, so speedup
    reflects the fused Triton kernel vs unfused PyTorch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gems_op = flag_gems._weight_int4pack_mm

    def set_shapes(self, shape_file_path=None):
        self.shapes = WEIGHT_INT4PACK_MM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from _weight_int4pack_mm_input_fn(shape, cur_dtype, self.device)

    def _run_metric(self, input_item):
        metric = BenchmarkMetrics()
        args = list(input_item)
        metric.shape_detail = self.record_shapes(*args)
        try:
            if "latency_base" in self.to_bench_metrics:
                metric.latency_base = self.get_latency(self.torch_op, *args)
            if "latency" in self.to_bench_metrics:
                metric.latency = self.get_latency(self.gems_op, *args)
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
        except (RuntimeError, Exception) as e:
            metric.error_msg = str(e)
            pytest.fail(str(e))
        return metric

    def run(self):
        if Config.query:
            self.init_default_config()
            from .consts import OperationAttribute

            attri = OperationAttribute(
                op_name=self.op_name,
                recommended_core_shapes=self.shapes,
                shape_desc="M,N,K,qGroupSize",
            )
            print(attri)
            emit_record_logger(attri.to_dict())
            return

        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            metrics = []
            input_iter = self.get_input_iter(dtype)
            done = False
            while not done:
                try:
                    input_item = next(input_iter)
                except StopIteration:
                    done = True
                    continue
                except (RuntimeError, Exception) as e:
                    print(
                        f"\033[31mFAILED\033[0m: Operator={self.op_name} "
                        f"dtype={dtype} err=<<<{e}>>>"
                    )
                    pytest.fail(str(e))

                metric = self._run_metric(input_item)
                metrics.append(metric)
                gc.collect()

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)
            update_result(self.op_name, result.to_json())
            emit_record_logger(result.to_json())


@pytest.mark.weight_int4pack_mm
def test_weight_int4pack_mm():
    bench = WeightInt4PackMmBenchmark(
        op_name="weight_int4pack_mm",
        torch_op=_torch_reference_int4pack_mm,
        dtypes=base.consts.FLOAT_DTYPES,
    )
    bench.run()
