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
from flag_gems.ops.bmm import bmm_out as triton_bmm_out
from flag_gems.ops.bmm_w8a8 import bmm_fp8_w8a8_block_scale

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
        yield inp1, inp2


@pytest.mark.bmm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_bmm(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="bmm",
        input_fn=_input_fn,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        inp2 = inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
    out = torch.empty([b, m, n], dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.bmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_bmm_out(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="bmm_out",
        input_fn=_input_fn_out,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def _is_fp8e4nv_supported():
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major + minor / 10 >= 8.9


def _quantize_a_fp8_per_mk_block(A, block_m=128, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, M, K = A.shape
    num_m_blocks = (M + block_m - 1) // block_m
    num_k_blocks = (K + block_k - 1) // block_k
    padded_m = num_m_blocks * block_m
    padded_k = num_k_blocks * block_k
    A_for_scale = A
    if padded_m != M:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros((batch, padded_m - M, K), dtype=A.dtype, device=A.device),
            ],
            dim=1,
        )
    if padded_k != K:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros(
                    (batch, padded_m, padded_k - K),
                    dtype=A.dtype,
                    device=A.device,
                ),
            ],
            dim=2,
        )
    A_blocked = A_for_scale.reshape(
        batch, num_m_blocks, block_m, num_k_blocks, block_k
    ).float()
    scale = (A_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    A_fp8 = (
        (A_blocked / scale[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    A_fp8 = A_fp8.reshape(batch, padded_m, padded_k)[:, :M, :K].contiguous()
    return A_fp8, scale.float().contiguous()


def _quantize_b_fp8_per_nk_block(B, block_n=128, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, K, N = B.shape
    num_k_blocks = (K + block_k - 1) // block_k
    num_n_blocks = (N + block_n - 1) // block_n
    padded_k = num_k_blocks * block_k
    padded_n = num_n_blocks * block_n
    B_for_scale = B
    if padded_k != K:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros((batch, padded_k - K, N), dtype=B.dtype, device=B.device),
            ],
            dim=1,
        )
    if padded_n != N:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros(
                    (batch, padded_k, padded_n - N),
                    dtype=B.dtype,
                    device=B.device,
                ),
            ],
            dim=2,
        )
    B_blocked = B_for_scale.reshape(
        batch, num_k_blocks, block_k, num_n_blocks, block_n
    ).float()
    scale = (B_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    B_fp8 = (
        (B_blocked / scale[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    B_fp8 = B_fp8.reshape(batch, padded_k, padded_n)[:, :K, :N].contiguous()
    return B_fp8, scale.float().contiguous()


def _triton_bmm_bf16_block_scale_baseline(
    A, B, A_fp8, B_fp8, A_scale, B_scale, out, block_m, block_n, block_k
):
    return triton_bmm_out(A, B, out)


def _gems_bmm_fp8_w8a8_block_scale(
    A, B, A_fp8, B_fp8, A_scale, B_scale, out, block_m, block_n, block_k
):
    return bmm_fp8_w8a8_block_scale(
        A_fp8,
        B_fp8,
        A_scale,
        B_scale,
        block_size=(block_m, block_n, block_k),
        out=out,
    )


class BmmFp8W8A8BlockScaleBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "B, M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 16, 32, 64),
            (4, 64, 64, 128),
            (2, 128, 128, 256),
            (4, 256, 256, 512),
            (8, 512, 512, 1024),
            (8, 1024, 1024, 2048),
            (8, 2048, 2048, 4096),
            (4, 4096, 4096, 16384),
            (8, 4096, 4096, 16384),
        ]

    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            A_fp8, A_scale = _quantize_a_fp8_per_mk_block(A)
            B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
            out = torch.empty((batch, m, n), device=self.device, dtype=dtype)
            yield A, B, A_fp8, B_fp8, A_scale, B_scale, out, 128, 128, 128


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 block-scale requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a8_block_scale_vs_triton_bf16():
    bench = BmmFp8W8A8BlockScaleBenchmark(
        op_name="bmm_fp8_w8a8_block_scale_vs_triton_bf16",
        torch_op=_triton_bmm_bf16_block_scale_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a8_block_scale)
    bench.run()
