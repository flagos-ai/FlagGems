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

import importlib
from types import SimpleNamespace

import pytest
import torch

import flag_gems

mm = importlib.import_module(flag_gems.mm.__module__)
pytestmark = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name != "thead"
    or not getattr(mm, "HAS_PPU_TLE", False),
    reason="requires the T-Head MM implementation",
)


@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "sliced", "singleton", "empty", "offset"]
)
def test_dense_transpose_detection_matches_torch(kind):
    import flag_gems

    mm = importlib.import_module(flag_gems.mm.__module__)
    candidates = {
        "dense": torch.empty(3, 4),
        "transpose": torch.empty(4, 3).t(),
        "sliced": torch.empty(8, 6)[::2, ::2],
        "singleton": torch.empty(1, 4).t(),
        "empty": torch.empty(0, 4).t(),
        "offset": torch.empty(20)[2:14].view(4, 3).t(),
    }
    value = candidates[kind]
    expected = (
        value.ndim == 2 and not value.is_contiguous() and value.t().is_contiguous()
    )
    assert mm._is_transposed_contiguous_2d(value) == expected


@pytest.mark.router_gemm
@pytest.mark.parametrize("nt", [False, True], ids=["nn", "nt"])
def test_fp16_router_gemm_uses_ppu(monkeypatch, nt):
    router = importlib.import_module(flag_gems.router_gemm.__module__)

    def reject_fallback(*args, **kwargs):
        pytest.fail("dense FP16 router GEMM fell back to the generic kernel")

    monkeypatch.setattr(router, "_generic_router_gemm", reject_fallback)
    x = torch.randn(1, 4096, dtype=torch.float16, device=flag_gems.device)
    b_shape = (64, 4096) if nt else (4096, 64)
    b = torch.randn(b_shape, dtype=x.dtype, device=x.device)
    weight = b if nt else b.t()
    out = flag_gems.router_gemm(x, weight)

    expected = x.cpu().float() @ weight.cpu().float().t()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-3)


@pytest.mark.mm
@pytest.mark.parametrize("case", ["column", "wide-row", "narrow-row"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_reuses_column_major_gemv(monkeypatch, case, dtype):
    device = flag_gems.device
    if case == "column":
        a = torch.randn(3073, 42, device=device, dtype=dtype)[:, 1:38].t()
        b = torch.randn(3073, 1, device=device, dtype=dtype)
        expected_configs = [(37, 3073, 42)]
    else:
        columns = 64 if case == "wide-row" else 8
        a = torch.randn(1, 4096, device=device, dtype=dtype)
        b = torch.randn(4096, columns, device=device, dtype=dtype)
        expected_configs = [(columns, 4096, columns)] if case == "wide-row" else []
    expected = (a.cpu().float() @ b.cpu().float()).to(dtype)
    calls = []
    config = mm._column_major_gemv_config

    def record_config(rows, k, lda):
        calls.append((rows, k, lda))
        return config(rows, k, lda)

    monkeypatch.setattr(mm, "_column_major_gemv_config", record_config)
    actual = flag_gems.mm(a, b)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.016, atol=0.04)
    assert calls == expected_configs
    if case == "column":
        out = a[:, :1]
        assert flag_gems.mm_out(a, b, out=out) is out
        torch.testing.assert_close(out.cpu(), expected, rtol=0.016, atol=0.04)


@pytest.mark.addmm
@pytest.mark.parametrize("beta", [0, 2])
def test_addmm_column_major_gemv_split_k(beta):
    device = flag_gems.device
    a = torch.randn(4096, 64, device=device, dtype=torch.bfloat16).t()
    b = torch.randn(4096, 1, device=device, dtype=a.dtype)
    bias = torch.full((64, 1), float("nan"), device=device, dtype=a.dtype)
    if beta != 0:
        bias.normal_()
    expected = torch.addmm(
        bias.cpu().float(), a.cpu().float(), b.cpu().float(), alpha=-0.5, beta=beta
    ).to(a.dtype)
    actual = flag_gems.addmm(bias, a, b, alpha=-0.5, beta=beta)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.016, atol=0.04)


def test_split_reducer_uses_this_launch_metadata_without_rebuilding_configs(
    monkeypatch,
):
    from contextlib import nullcontext

    import flag_gems

    mm = importlib.import_module(flag_gems.mm.__module__)
    selected = [2, 4, 2]
    seen = []

    class Main:
        best_config = SimpleNamespace(kwargs={"SPLIT_K": 8})

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return self, {"SPLIT_K": selected.pop(0)}

            return launch

    class Reduce:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.append(kwargs["SPLIT_K"])

            return launch

    def rebuild():
        raise AssertionError("candidate catalog rebuilt on hot path")

    monkeypatch.setattr(mm, "mm_split_k_kernel_ppu", Main())
    monkeypatch.setattr(mm, "mm_split_k_reduce_kernel_ppu", Reduce())
    monkeypatch.setattr(mm, "_ppu_split_k_configs", rebuild)
    monkeypatch.setattr(mm, "_ppu_device_context", lambda tensor: nullcontext())
    a = torch.empty(4, 2048, dtype=torch.bfloat16)
    b = torch.empty(2048, 64, dtype=torch.bfloat16)
    out = torch.empty(4, 64, dtype=torch.bfloat16)
    for _ in range(3):
        assert mm._run_split_k_mm(a, b, out) is out
    assert seen == [2, 4, 2]
