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
import triton

import flag_gems

from . import base, consts

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
BLOCK_SIZE = (128, 128, 128)


def _cuda_fp8_available():
    if FP8_DTYPE is None or flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def _quantize_a(A, block_m, block_k):
    M, K = A.shape
    padded_m = triton.cdiv(M, block_m) * block_m
    padded_k = triton.cdiv(K, block_k) * block_k
    padded = torch.zeros((padded_m, padded_k), dtype=A.dtype, device=A.device)
    padded[:M, :K] = A
    grouped = padded.reshape(
        padded_m // block_m, block_m, padded_k // block_k, block_k
    ).float()
    fp8_info = torch.finfo(FP8_DTYPE)
    scale = (grouped.abs().amax(dim=(1, 3)) / fp8_info.max).clamp(min=1e-8)
    quantized = (
        (grouped / scale[:, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
        .reshape(padded_m, padded_k)[:M, :K]
        .contiguous()
    )
    return quantized, scale.float().contiguous()


def _quantize_b(B, block_n, block_k):
    groups, K, N = B.shape
    padded_k = triton.cdiv(K, block_k) * block_k
    padded_n = triton.cdiv(N, block_n) * block_n
    padded = torch.zeros((groups, padded_n, padded_k), dtype=B.dtype, device=B.device)
    padded[:, :N, :K] = B.transpose(-1, -2)
    grouped = padded.reshape(
        groups, padded_n // block_n, block_n, padded_k // block_k, block_k
    ).float()
    fp8_info = torch.finfo(FP8_DTYPE)
    scale_nk = (grouped.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    quantized_nk = (
        (grouped / scale_nk[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
        .reshape(groups, padded_n, padded_k)[:, :N, :K]
        .contiguous()
    )
    scale = scale_nk.permute(0, 2, 1).float().contiguous()
    return quantized_nk.transpose(-1, -2), scale


def _bf16_group_mm(A, B, A_fp8, B_fp8, A_scale, B_scale, offs, block_size):
    return flag_gems.group_mm(A, B, offs)


def _fp8_group_mm(A, B, A_fp8, B_fp8, A_scale, B_scale, offs, block_size):
    return flag_gems.group_gemm_w8a8_fp8(
        A_fp8, B_fp8, A_scale, B_scale, offs, block_size=block_size
    )


class GroupGemmW8A8Fp8Benchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_SHAPE_DESC = "groups, M_per_group, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (16, 8, 512, 2048),
            (16, 32, 2048, 2048),
            (16, 128, 2048, 2048),
            (64, 4, 2048, 128),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [(8, 512, 4096, 4096)]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        block_m, block_n, block_k = BLOCK_SIZE
        for groups, m_per_group, N, K in self.shapes:
            M = groups * m_per_group
            A = torch.randn((M, K), dtype=dtype, device=self.device) * 0.25
            B = torch.randn((groups, K, N), dtype=dtype, device=self.device) * 0.25
            A_fp8, A_scale = _quantize_a(A, block_m, block_k)
            B_fp8, B_scale = _quantize_b(B, block_n, block_k)
            offs = torch.arange(
                m_per_group,
                M + 1,
                m_per_group,
                dtype=torch.int32,
                device=self.device,
            )
            yield A, B, A_fp8, B_fp8, A_scale, B_scale, offs, BLOCK_SIZE

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[:2]
        return 2 * A.shape[0] * B.shape[1] * B.shape[2]


@pytest.mark.group_gemm_w8a8_fp8
@pytest.mark.skipif(
    not _cuda_fp8_available(),
    reason="group GEMM W8A8 FP8 benchmark requires CUDA SM90+ FP8 support",
)
def test_group_gemm_w8a8_fp8_benchmark():
    benchmark = GroupGemmW8A8Fp8Benchmark(
        op_name="group_gemm_w8a8_fp8",
        torch_op=_bf16_group_mm,
        dtypes=[torch.bfloat16],
    )
    benchmark.set_gems(_fp8_group_mm)
    benchmark.run()
