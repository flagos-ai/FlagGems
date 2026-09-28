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

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

pytestmark = pytest.mark.skipif(
    not hasattr(flag_gems, "scaled_grouped_mm"),
    reason="flag_gems.scaled_grouped_mm is unavailable",
)


if QUICK_MODE:
    CASES = ["m_varying"]
else:
    CASES = ["m_varying", "batch", "k_varying", "n_varying"]


def _float8_dtypes():
    names = ["float8_e4m3fn"]
    if torch.version.hip:
        names.append("float8_e4m3fnuz")
    return [getattr(torch, name) for name in names if hasattr(torch, name)]


def _is_float8(dtype):
    return dtype in _float8_dtypes()


def _dtype_cases():
    dtypes = [torch.float16, torch.float32]
    if flag_gems.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    dtypes.extend(_float8_dtypes())
    return dtypes


def _default_out_dtype(dtype):
    if _is_float8(dtype):
        return torch.bfloat16
    return dtype


def _make_tensor(shape, dtype):
    # Quantize on CPU; storage transfer does not require a vendor FP8 cast kernel.
    return (
        (torch.randn(shape, dtype=torch.float32) * 0.25).to(dtype).to(flag_gems.device)
    )


def _make_case(case_name, dtype):
    groups, M, N, K = 3, 5, 16, 32

    if case_name == "m_varying":
        sizes = [2, 3, 4]
        total_m = sum(sizes)
        mat_a = _make_tensor((total_m, K), dtype)
        mat_b = _make_tensor((groups, K, N), dtype)
        offs = torch.tensor(
            [sum(sizes[: i + 1]) for i in range(groups)],
            dtype=torch.int32,
            device=flag_gems.device,
        )
        scale_a = torch.linspace(0.75, 1.25, total_m, device="cpu").to(flag_gems.device)
        scale_b = torch.linspace(1.25, 0.75, groups * N, device="cpu").to(
            flag_gems.device
        )
        return mat_a, mat_b, scale_a, scale_b.reshape(groups, N), offs

    if case_name == "batch":
        mat_a = _make_tensor((groups, M, K), dtype)
        mat_b = _make_tensor((groups, K, N), dtype)
        scale_a = torch.linspace(0.75, 1.25, groups * M, device="cpu").to(
            flag_gems.device
        )
        scale_b = torch.linspace(1.25, 0.75, groups * N, device="cpu").to(
            flag_gems.device
        )
        return (
            mat_a,
            mat_b,
            scale_a.reshape(groups, M),
            scale_b.reshape(groups, N),
            None,
        )

    if case_name == "k_varying":
        k_offsets = [8, 20, K]
        mat_a = _make_tensor((M, K), dtype)
        mat_b = _make_tensor((K, N), dtype)
        offs = torch.tensor(k_offsets, dtype=torch.int32, device="cpu").to(
            flag_gems.device
        )
        scale_a = torch.linspace(0.75, 1.25, groups * M, device="cpu").to(
            flag_gems.device
        )
        scale_b = torch.linspace(1.25, 0.75, groups * N, device="cpu").to(
            flag_gems.device
        )
        return mat_a, mat_b, scale_a, scale_b, offs

    n_offsets = [8, 20, 32]
    mat_a = _make_tensor((groups, M, K), dtype)
    mat_b = _make_tensor((K, n_offsets[-1]), dtype)
    offs = torch.tensor(n_offsets, dtype=torch.int32, device="cpu").to(flag_gems.device)
    scale_a = torch.linspace(0.75, 1.25, groups * M, device="cpu").to(flag_gems.device)
    scale_b = torch.linspace(1.25, 0.75, n_offsets[-1], device="cpu").to(
        flag_gems.device
    )
    return mat_a, mat_b, scale_a.reshape(groups, M), scale_b, offs


def _make_bias(case_name, mat_b, groups, use_bias, out_dtype):
    if not use_bias:
        return None

    if case_name == "n_varying":
        return torch.randn((mat_b.shape[-1],), dtype=torch.float32, device="cpu").to(
            flag_gems.device
        )
    if case_name in ("batch", "k_varying"):
        return torch.randn(
            (groups, mat_b.shape[-1]), dtype=torch.float32, device="cpu"
        ).to(flag_gems.device)
    return torch.randn((mat_b.shape[-1],), dtype=torch.float32, device="cpu").to(
        flag_gems.device
    )


def _scale_and_bias(out, scale_a, scale_b, bias, out_dtype):
    out = out * scale_a * scale_b
    if bias is not None:
        out = out + bias
    return out.to(out_dtype)


def _reference(mat_a, mat_b, scale_a, scale_b, offs, bias, out_dtype):
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    groups = offs.numel() if a_is_2d and b_is_2d else mat_a.shape[0]
    if a_is_2d and not b_is_2d:
        groups = mat_b.shape[0]

    mat_a = mat_a.detach().cpu().float()
    mat_b = mat_b.detach().cpu().float()
    scale_a = scale_a.detach().cpu().float()
    scale_b = scale_b.detach().cpu().float()
    bias = bias.detach().cpu().float() if bias is not None else None
    offsets = [0] + (offs.detach().cpu().tolist() if offs is not None else [])

    chunks = []
    if a_is_2d and not b_is_2d:
        for group_idx in range(groups):
            m_start, m_end = offsets[group_idx], offsets[group_idx + 1]
            chunk = mat_a[m_start:m_end].mm(mat_b[group_idx])
            chunk_bias = bias if bias is None or bias.dim() == 1 else bias[group_idx]
            chunks.append(
                _scale_and_bias(
                    chunk,
                    scale_a[m_start:m_end].reshape(-1, 1),
                    scale_b[group_idx].reshape(1, -1),
                    chunk_bias,
                    out_dtype,
                )
            )
        return torch.cat(chunks, dim=0)

    if not a_is_2d and b_is_2d:
        for group_idx in range(groups):
            n_start, n_end = offsets[group_idx], offsets[group_idx + 1]
            chunk = mat_a[group_idx].mm(mat_b[:, n_start:n_end])
            chunk_bias = bias[n_start:n_end] if bias is not None else None
            chunks.append(
                _scale_and_bias(
                    chunk,
                    scale_a[group_idx].reshape(-1, 1),
                    scale_b[n_start:n_end].reshape(1, -1),
                    chunk_bias,
                    out_dtype,
                )
            )
        return torch.cat(chunks, dim=1)

    if a_is_2d and b_is_2d:
        scale_a = scale_a.reshape(groups, mat_a.shape[0])
        scale_b = scale_b.reshape(groups, mat_b.shape[1])
        for group_idx in range(groups):
            k_start, k_end = offsets[group_idx], offsets[group_idx + 1]
            chunk = mat_a[:, k_start:k_end].mm(mat_b[k_start:k_end])
            chunk_bias = bias if bias is None or bias.dim() == 1 else bias[group_idx]
            chunks.append(
                _scale_and_bias(
                    chunk,
                    scale_a[group_idx].reshape(-1, 1),
                    scale_b[group_idx].reshape(1, -1),
                    chunk_bias,
                    out_dtype,
                )
            )
        return torch.stack(chunks, dim=0)

    for group_idx in range(groups):
        chunk = mat_a[group_idx].mm(mat_b[group_idx])
        chunk_bias = bias if bias is None or bias.dim() == 1 else bias[group_idx]
        chunks.append(
            _scale_and_bias(
                chunk,
                scale_a[group_idx].reshape(-1, 1),
                scale_b[group_idx].reshape(1, -1),
                chunk_bias,
                out_dtype,
            )
        )
    return torch.stack(chunks, dim=0)


@pytest.mark.scaled_grouped_mm
@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("dtype", _dtype_cases(), ids=str)
@pytest.mark.parametrize("use_bias", [False, True])
def test_scaled_grouped_mm(case_name, dtype, use_bias):
    mat_a, mat_b, scale_a, scale_b, offs = _make_case(case_name, dtype)
    groups = offs.numel() if offs is not None else mat_a.shape[0]
    out_dtype = torch.float32 if dtype == torch.float16 and use_bias else None
    target_dtype = out_dtype or _default_out_dtype(dtype)
    bias = _make_bias(case_name, mat_b, groups, use_bias, target_dtype)

    ref = _reference(mat_a, mat_b, scale_a, scale_b, offs, bias, target_dtype)
    res = flag_gems.scaled_grouped_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs=offs,
        bias=bias,
        out_dtype=out_dtype,
        use_fast_accum=True,
    )

    assert res.dtype == target_dtype
    ref = ref if utils.TO_CPU else ref.to(flag_gems.device)

    if _is_float8(dtype):
        res = res.cpu() if utils.TO_CPU else res
        torch.testing.assert_close(
            res.float(),
            ref.float(),
            atol=2.5e-1,
            rtol=5e-1,
        )
    else:
        utils.gems_assert_close(res, ref, target_dtype, reduce_dim=mat_a.shape[-1])


@pytest.mark.scaled_grouped_mm
@pytest.mark.parametrize("dtype", _float8_dtypes(), ids=str)
@pytest.mark.parametrize(
    "groups, m_per_group, N, K",
    [(4, 64, 128, 128), (8, 128, 256, 256), (16, 256, 512, 512)],
)
def test_scaled_grouped_mm_fp8_core(groups, m_per_group, N, K, dtype):
    sizes = [m_per_group + group for group in range(groups)]
    offsets = [sum(sizes[: group + 1]) for group in range(groups)]
    M = offsets[-1]
    # Retain CPU inputs: copying a transposed FP8 tensor back from the device
    # may require native stride/cast support unrelated to the Gems kernel.
    a_cpu = (torch.randn((M, K), dtype=torch.float32) * 0.25).to(dtype)
    b_cpu = (torch.randn((groups, N, K), dtype=torch.float32) * 0.25).to(dtype)
    offs_cpu = torch.tensor(offsets, dtype=torch.int32, device="cpu")
    scale_a_cpu = torch.linspace(0.75, 1.25, M, device="cpu")
    scale_b_cpu = torch.linspace(1.25, 0.75, groups * N, device="cpu").reshape(
        groups, N
    )
    ref = _reference(
        a_cpu,
        b_cpu.transpose(-1, -2),
        scale_a_cpu,
        scale_b_cpu,
        offs_cpu,
        None,
        torch.bfloat16,
    )
    mat_a = a_cpu.to(flag_gems.device)
    mat_b = b_cpu.to(flag_gems.device).transpose(-1, -2)
    offs = offs_cpu.to(flag_gems.device)
    scale_a = scale_a_cpu.to(flag_gems.device)
    scale_b = scale_b_cpu.to(flag_gems.device)
    res = flag_gems.scaled_grouped_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs=offs,
        bias=None,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    assert res.dtype == torch.bfloat16
    # Preserve the existing scaled_grouped_mm FP8 error budget while extending
    # coverage; tighten it only with device accuracy evidence, not timing results.
    torch.testing.assert_close(res.cpu().float(), ref.float(), atol=2.5e-1, rtol=5e-1)


@pytest.mark.scaled_grouped_mm
@pytest.mark.parametrize("dtype", _float8_dtypes(), ids=str)
def test_scaled_grouped_mm_fp8_encodings(dtype):
    # Cover every finite encoding, including subnormals and the largest values.
    bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    if dtype == getattr(torch, "float8_e4m3fnuz", None):
        bits[bits == 128] = 0
    else:
        bits[(bits & 127) == 127] = 0
    a_cpu = bits.view(dtype).reshape(1, 16, 16)
    b_cpu = torch.eye(16).to(dtype).reshape(1, 16, 16)
    scales = torch.ones((1, 16), dtype=torch.float32).to(flag_gems.device)
    result = flag_gems.scaled_grouped_mm(
        a_cpu.to(flag_gems.device),
        b_cpu.to(flag_gems.device),
        scales,
        scales,
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(result.cpu(), a_cpu.float(), atol=0, rtol=0)


@pytest.mark.scaled_grouped_mm
def test_scaled_grouped_mm_scale_before_output_cast():
    # The unscaled product is 65536: casting it to FP16 before scaling overflows.
    mat_a = torch.full((1, 3, 16), 64.0, dtype=torch.float16, device="cpu").to(
        flag_gems.device
    )
    mat_b = torch.full((1, 16, 5), 64.0, dtype=torch.float16, device="cpu").to(
        flag_gems.device
    )
    scale_a = torch.full((1, 3), 0.5, dtype=torch.float32, device="cpu").to(
        flag_gems.device
    )
    scale_b = torch.full((1, 5), 0.5, dtype=torch.float32, device="cpu").to(
        flag_gems.device
    )
    result = flag_gems.scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b)
    expected = torch.full((1, 3, 5), 16384.0, dtype=torch.float16)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.scaled_grouped_mm
@pytest.mark.parametrize("transposed_b", [False, True])
def test_scaled_grouped_mm_reduction_addressing(transposed_b):
    # Select K entries in three different tiles, including the padded tail.
    # All values are binary fractions, so the expected FP32 result is exact.
    k, n = 65, 5
    selected = [0, 32, 64]
    a_cpu = torch.zeros((1, len(selected), k), dtype=torch.float16)
    for row, index in enumerate(selected):
        a_cpu[0, row, index] = 1
    b_cpu = (torch.arange(k * n).reshape(1, k, n) % 17).to(torch.float16) / 8
    if transposed_b:
        b_cpu = b_cpu.transpose(-1, -2).contiguous().transpose(-1, -2)
    scale_a = torch.tensor([[0.5, 1.0, 2.0]], dtype=torch.float32)
    scale_b = torch.tensor([[0.5, 1.0, 2.0, 0.5, 1.0]], dtype=torch.float32)
    expected = b_cpu[:, selected, :].float() * scale_a[:, :, None] * scale_b[:, None, :]
    result = flag_gems.scaled_grouped_mm(
        a_cpu.to(flag_gems.device),
        b_cpu.to(flag_gems.device),
        scale_a.to(flag_gems.device),
        scale_b.to(flag_gems.device),
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
