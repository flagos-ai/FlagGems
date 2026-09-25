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


# One-time weight-conversion cache. The ATen int4 GEMM consumes a repacked
# (tiled) weight, not this operator's raw byte-pair layout; converting it is a
# one-off deployment step, not part of per-call inference. do_bench calls the
# baseline many times with the same input tensors, so we key the converted
# weight (and reparametrized scale/zero) on the packed tensor's identity and
# reuse it across timed iterations -- only the ATen matmul is measured.
_ATEN_CONVERT_CACHE = {}


def _aten_inner_k_tiles(K):
    """Largest innerKTiles in {8,4,2} satisfying ATen's K % (ikt*16) == 0."""
    for ikt in (8, 4, 2):
        if K % (ikt * 16) == 0:
            return ikt
    return 2


def _torch_reference_int4pack_mm(A, mat2_packed, qGroupSize, qScaleAndZeros):
    """ATen ``_weight_int4pack_mm`` baseline for the speedup measurement.

    This dispatches to the *real* ATen operator this PR implements, per the
    reviewer's request to use ``torch.ops.aten._weight_int4pack_mm`` as the
    torch baseline:

    - bf16 activations -> the CUDA overload ``torch.ops.aten._weight_int4pack_mm``
      runs on the same device, so the speedup ratio is a fair GPU-vs-GPU
      comparison against NVIDIA's tiled int4 GEMM.
    - fp16 / fp32 activations -> the CUDA overload only accepts bf16, so we fall
      back to the CPU overload ``torch.ops.aten._weight_int4pack_mm_for_cpu``
      (the same real ATen op, CPU variant).

    Both overloads consume their own repacked weight built from this operator's
    byte-pair layout, and both dequantize as ``w = scale*(q - 8) + zero_add``;
    this operator uses ``w = (q - zero)*scale``, so we reparametrize
    ``zero_add = scale*(8 - zero)`` to make the two identical.
    """
    N, K_half = mat2_packed.shape
    K = K_half * 2
    scale = qScaleAndZeros[:, :, 0]
    zero = qScaleAndZeros[:, :, 1]

    if A.dtype == torch.bfloat16:
        key = (mat2_packed.data_ptr(), qGroupSize, "cuda")
        cached = _ATEN_CONVERT_CACHE.get(key)
        if cached is None:
            # This op packs low nibble = even column, high nibble = odd column.
            # ATen's packer expects the opposite nibble order, so swap them.
            low = mat2_packed & 0xF
            high = (mat2_packed >> 4) & 0xF
            aten_packed = ((low << 4) | high).contiguous().to(torch.uint8)
            weight = torch.ops.aten._convert_weight_to_int4pack(
                aten_packed, _aten_inner_k_tiles(K)
            )
            zero_add = (scale * (8.0 - zero)).to(A.dtype)
            sz = torch.stack([scale.to(A.dtype), zero_add], dim=-1).contiguous()
            cached = (weight, sz)
            _ATEN_CONVERT_CACHE[key] = cached
        weight, sz = cached
        return torch.ops.aten._weight_int4pack_mm(A, weight, qGroupSize, sz)

    # fp16 / fp32: CUDA overload rejects non-bf16, use the CPU overload.
    key = (mat2_packed.data_ptr(), qGroupSize, "cpu")
    cached = _ATEN_CONVERT_CACHE.get(key)
    if cached is None:
        mp = mat2_packed.cpu()
        low = (mp & 0xF).to(torch.int32)
        high = ((mp >> 4) & 0xF).to(torch.int32)
        q = torch.empty((N, K), dtype=torch.int32)
        q[:, 0::2] = low
        q[:, 1::2] = high
        weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(q, 2)
        scale_cpu = scale.cpu().float()
        zero_add = scale_cpu * (8.0 - zero.cpu().float())
        sz = torch.stack([scale_cpu, zero_add], dim=-1).contiguous()
        cached = (weight, sz)
        _ATEN_CONVERT_CACHE[key] = cached
    weight, sz = cached
    out = torch.ops.aten._weight_int4pack_mm_for_cpu(
        A.cpu().float(), weight, qGroupSize, sz
    )
    return out.to(A.device).to(A.dtype)


def _weight_int4pack_mm_input_fn(shape, dtype, device):
    """Yield input tuples for _weight_int4pack_mm benchmark."""
    M, N, K, qGroupSize = shape
    # Create activation tensor
    A = torch.randn((M, K), dtype=dtype, device=device)
    # Create int4 weights (values 0..15)
    weight_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    # Pack weights into byte-pair format: low nibble = even col, high = odd col.
    even = (weight_int4[:, 0::2] & 0xF).to(torch.uint8)
    odd = (weight_int4[:, 1::2] & 0xF).to(torch.uint8)
    packed = ((odd << 4) | even).contiguous()
    # Create scales and zeros
    num_groups = K // qGroupSize
    scales = torch.rand((num_groups, N), dtype=dtype, device=device) * 1.5 + 0.5
    zeros = torch.randint(4, 11, (num_groups, N), dtype=dtype, device=device)
    qScaleAndZeros = torch.stack([scales, zeros], dim=-1)
    yield A, packed, qGroupSize, qScaleAndZeros


class WeightInt4PackMmBenchmark(base.Benchmark):
    """Benchmark for _weight_int4pack_mm operator.

    The timing baseline is the real ATen operator this PR implements
    (``torch.ops.aten._weight_int4pack_mm`` for bf16, its ``_for_cpu`` overload
    for fp16/fp32); see ``_torch_reference_int4pack_mm`` for the packing/dtype
    handling. Speedup therefore compares the fused Triton kernel against the
    genuine ATen int4 GEMM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gems_op = flag_gems.weight_int4pack_mm

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
