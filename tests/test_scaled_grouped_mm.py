import pytest
import torch
from packaging import version

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    CASES = ["m_varying"]
else:
    CASES = ["m_varying", "batch", "k_varying", "n_varying"]


def _cuda_fp8_available():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


def _float8_dtypes():
    return [torch.float8_e4m3fn] if _cuda_fp8_available() else []


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
    base = torch.randn(shape, dtype=torch.float32, device=flag_gems.device) * 0.25
    try:
        return base.to(dtype)
    except RuntimeError as exc:
        pytest.skip(f"{dtype} is not supported on {flag_gems.device}: {exc}")


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
        scale_a = torch.linspace(0.75, 1.25, total_m, device=flag_gems.device)
        scale_b = torch.linspace(1.25, 0.75, groups * N, device=flag_gems.device)
        return mat_a, mat_b, scale_a, scale_b.reshape(groups, N), offs

    if case_name == "batch":
        mat_a = _make_tensor((groups, M, K), dtype)
        mat_b = _make_tensor((groups, K, N), dtype)
        scale_a = torch.linspace(0.75, 1.25, groups * M, device=flag_gems.device)
        scale_b = torch.linspace(1.25, 0.75, groups * N, device=flag_gems.device)
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
        offs = torch.tensor(k_offsets, dtype=torch.int32, device=flag_gems.device)
        scale_a = torch.linspace(0.75, 1.25, groups * M, device=flag_gems.device)
        scale_b = torch.linspace(1.25, 0.75, groups * N, device=flag_gems.device)
        return mat_a, mat_b, scale_a, scale_b, offs

    n_offsets = [8, 20, 32]
    mat_a = _make_tensor((groups, M, K), dtype)
    mat_b = _make_tensor((K, n_offsets[-1]), dtype)
    offs = torch.tensor(n_offsets, dtype=torch.int32, device=flag_gems.device)
    scale_a = torch.linspace(0.75, 1.25, groups * M, device=flag_gems.device)
    scale_b = torch.linspace(1.25, 0.75, n_offsets[-1], device=flag_gems.device)
    return mat_a, mat_b, scale_a.reshape(groups, M), scale_b, offs


def _make_bias(case_name, mat_b, groups, use_bias, out_dtype):
    if not use_bias:
        return None

    if case_name == "n_varying":
        return torch.randn(
            (mat_b.shape[-1],), dtype=torch.float32, device=flag_gems.device
        )
    if case_name in ("batch", "k_varying"):
        return torch.randn(
            (groups, mat_b.shape[-1]), dtype=torch.float32, device=flag_gems.device
        )
    return torch.randn((mat_b.shape[-1],), dtype=torch.float32, device=flag_gems.device)


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
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.8"),
    reason="aten._scaled_grouped_mm requires PyTorch >= 2.8.0.",
)
@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("dtype", _dtype_cases())
@pytest.mark.parametrize("use_bias", [False, True])
def test_scaled_grouped_mm(case_name, dtype, use_bias):
    mat_a, mat_b, scale_a, scale_b, offs = _make_case(case_name, dtype)
    groups = offs.numel() if offs is not None else mat_a.shape[0]
    out_dtype = torch.float32 if dtype == torch.float16 and use_bias else None
    target_dtype = out_dtype or _default_out_dtype(dtype)
    bias = _make_bias(case_name, mat_b, groups, use_bias, target_dtype)

    ref = _reference(mat_a, mat_b, scale_a, scale_b, offs, bias, target_dtype)
    with flag_gems.use_gems():
        res = torch._scaled_grouped_mm(
            mat_a,
            mat_b,
            scale_a,
            scale_b,
            offs=offs,
            bias=bias,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

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
