# Copyright 2026 XCoreSigma Contributors
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
from unittest.mock import Mock

import pytest
import torch

import flag_gems

from . import conftest as cfg

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

ascend_only = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="the optimized Grouped MatMul implementation targets Ascend",
)

GROUPED_MATMUL_FULL_CASES = (
    (1024, 2048, 4096, 32),
    (1024, 4096, 1024, 32),
    (2560, 2048, 4096, 32),
    (2560, 4096, 1024, 32),
    (3339, 4096, 1024, 32),
    (7354, 4096, 1024, 32),
    (7354, 2048, 4096, 32),
    (4096, 4096, 2048, 32),
)
GROUPED_MATMUL_CASES = (
    (
        GROUPED_MATMUL_FULL_CASES[0],
        GROUPED_MATMUL_FULL_CASES[5],
        GROUPED_MATMUL_FULL_CASES[7],
    )
    if cfg.QUICK_MODE
    else GROUPED_MATMUL_FULL_CASES
)


def _make_group_list(m, groups, group_list_type, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    boundaries = torch.floor(torch.rand(groups - 1, generator=generator) * m)
    cumulative = torch.cat(
        (
            torch.sort(boundaries)[0].to(torch.int64),
            torch.tensor([m], dtype=torch.int64),
        )
    )
    if group_list_type == 0:
        return cumulative.to(device=flag_gems.device)
    counts = cumulative.clone()
    counts[1:] -= cumulative[:-1]
    return counts.to(device=flag_gems.device)


def _make_matmul_inputs(case, group_list_type, dtype=torch.float16):
    m, n, k, groups = case
    case_index = GROUPED_MATMUL_FULL_CASES.index(case)
    torch.manual_seed(1000 + case_index)
    x = torch.rand((m, k), device=flag_gems.device, dtype=dtype)
    weight = torch.rand((groups, k, n), device=flag_gems.device, dtype=dtype)
    group_list = _make_group_list(
        m,
        groups,
        group_list_type,
        2026 + case_index,
    )
    return x, weight, group_list


def _official(x, weight, group_list, group_list_type):
    return torch_npu.npu_grouped_matmul(
        [x],
        [weight],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
    )[0]


@pytest.mark.grouped_matmul
@ascend_only
@pytest.mark.parametrize("case", GROUPED_MATMUL_CASES)
@pytest.mark.parametrize("group_list_type", (0, 1))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_grouped_matmul_accuracy(case, group_list_type, dtype):
    x, weight, group_list = _make_matmul_inputs(case, group_list_type, dtype)
    expected = _official(x, weight, group_list, group_list_type)
    actual = flag_gems.grouped_matmul(
        x,
        weight,
        group_list,
        group_list_type,
    )
    torch.npu.synchronize()
    # FP32 reduction orders can round midpoint values to adjacent BF16 numbers.
    atol, rtol = (
        (1.0, 2e-3) if dtype == torch.float16 else (0.0, torch.finfo(dtype).eps)
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.fixture
def grouped_matmul_module():
    return importlib.import_module(flag_gems.grouped_matmul.__module__)


@pytest.mark.grouped_matmul
@ascend_only
@pytest.mark.parametrize(
    "kernel_name", ("_grouped_matmul_nd_kernel", "_grouped_matmul_panel_kernel")
)
@pytest.mark.parametrize("group_list_type", (0, 1))
def test_grouped_matmul_large_weight_offset(
    grouped_matmul_module, kernel_name, group_list_type
):
    groups, k, n = 151, 2048, 7168
    assert (groups - 1) * k * n > 2**31
    free_bytes, _ = torch.npu.mem_get_info()
    if free_bytes < groups * k * n * 2 + 2 * 1024**3:
        pytest.skip("large-offset regression requires 6.2 GiB of free NPU memory")

    x = torch.randn((1, k), device=flag_gems.device, dtype=torch.float16) * 0.1
    weight = torch.empty((groups, k, n), device=flag_gems.device, dtype=x.dtype)
    # Only the final group has rows; the preceding weights must never be read.
    weight[-1].normal_(std=0.1)
    group_list = torch.zeros(groups, device=flag_gems.device, dtype=torch.int64)
    group_list[-1] = 1
    expected = _official(x, weight, group_list, group_list_type)
    actual = torch.empty((1, n), device=flag_gems.device, dtype=x.dtype)
    kernel = getattr(grouped_matmul_module, kernel_name)
    kernel[(grouped_matmul_module._get_aic_core_num(x.device.index),)](
        x,
        weight,
        actual,
        group_list,
        GROUPS=groups,
        N=n,
        K=k,
        GROUP_LIST_TYPE=group_list_type,
        multibuffer=True,
        unit_flag=True,
        sync_solver=False,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.grouped_matmul
@ascend_only
@pytest.mark.parametrize("cores", (1, 24, 32))
def test_grouped_matmul_core_api(grouped_matmul_module, monkeypatch, cores):
    query = Mock(return_value={"cube_core_num": cores, "vector_core_num": cores * 2})
    monkeypatch.setattr(
        grouped_matmul_module.torch_device_fn, "get_device_limit", query, raising=False
    )
    assert grouped_matmul_module._get_aic_core_num(3) == cores
    query.assert_called_once_with(3)


@pytest.mark.grouped_matmul
@ascend_only
@pytest.mark.parametrize(
    "limits",
    (
        {},
        None,
        {"cube_core_num": 0},
        {"cube_core_num": -1},
        {"cube_core_num": None},
        {"cube_core_num": True},
    ),
)
def test_grouped_matmul_invalid_core_limit(grouped_matmul_module, monkeypatch, limits):
    monkeypatch.setattr(
        grouped_matmul_module.torch_device_fn,
        "get_device_limit",
        Mock(return_value=limits),
        raising=False,
    )
    assert grouped_matmul_module._get_aic_core_num(0) == 24


@pytest.mark.grouped_matmul
@ascend_only
@pytest.mark.parametrize("error", (AttributeError, RuntimeError))
def test_grouped_matmul_core_api_unavailable(grouped_matmul_module, monkeypatch, error):
    monkeypatch.setattr(
        grouped_matmul_module.torch_device_fn,
        "get_device_limit",
        Mock(side_effect=error("unavailable")),
        raising=False,
    )
    assert grouped_matmul_module._get_aic_core_num(0) == 24


@pytest.mark.grouped_matmul
@ascend_only
@pytest.mark.parametrize("num_cores", (1, 3))
@pytest.mark.parametrize("group_list_type", (0, 1))
@pytest.mark.parametrize("counts", ((0, 1, 0, 127, 128, 129, 0), (0, 0), (1,)))
def test_grouped_matmul_group_boundaries(
    grouped_matmul_module, monkeypatch, num_cores, group_list_type, counts
):
    monkeypatch.setattr(grouped_matmul_module, "_get_aic_core_num", lambda _: num_cores)
    m = sum(counts)
    x = torch.randn((m, 256), device=flag_gems.device, dtype=torch.float16)
    weight = torch.randn(
        (len(counts), 256, 256), device=flag_gems.device, dtype=torch.float16
    )
    group_list = torch.tensor(counts, dtype=torch.int32)
    if group_list_type == 0:
        group_list = group_list.cumsum(0, dtype=torch.int32)
    group_list = group_list.to(flag_gems.device)
    actual = flag_gems.grouped_matmul(x, weight, group_list, group_list_type)
    if m:
        expected = _official(x, weight, group_list.to(torch.int64), group_list_type)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=2e-3)
    else:
        torch.npu.synchronize()
        assert actual.shape == (0, 256)
        assert actual.dtype == x.dtype


@pytest.mark.grouped_matmul
@ascend_only
def test_grouped_matmul_rejects_invalid_group_list_type():
    x = torch.empty((1, 256), device=flag_gems.device, dtype=torch.float16)
    weight = torch.empty((1, 256, 256), device=flag_gems.device, dtype=torch.float16)
    group_list = torch.ones((1,), device=flag_gems.device, dtype=torch.int64)
    with pytest.raises(ValueError, match="group_list_type"):
        flag_gems.grouped_matmul(x, weight, group_list, 2)


@pytest.mark.grouped_matmul
@ascend_only
@pytest.mark.parametrize(
    "field,shape,dtype,error,match",
    (
        ("x", (256,), torch.float16, ValueError, "x must have shape"),
        ("x", (1, 256), torch.float32, TypeError, "float16 and bfloat16"),
        ("weight", (256, 256), torch.float16, ValueError, "weight must have shape"),
        ("weight", (1, 128, 256), torch.float16, ValueError, "same K"),
        ("weight", (1, 256, 256), torch.bfloat16, TypeError, "same dtype"),
        ("weight", (1, 256, 0), torch.float16, ValueError, "positive"),
        ("group_list", (1, 1), torch.int64, ValueError, "1D tensor"),
        ("group_list", (2,), torch.int64, ValueError, "one value per group"),
        ("group_list", (1,), torch.float32, TypeError, "int32 or int64"),
    ),
)
def test_grouped_matmul_rejects_invalid_metadata(field, shape, dtype, error, match):
    inputs = {
        "x": torch.empty((1, 256), device=flag_gems.device, dtype=torch.float16),
        "weight": torch.empty(
            (1, 256, 256), device=flag_gems.device, dtype=torch.float16
        ),
        "group_list": torch.empty((1,), device=flag_gems.device, dtype=torch.int64),
    }
    inputs[field] = torch.empty(shape, device=flag_gems.device, dtype=dtype)
    with pytest.raises(error, match=match):
        flag_gems.grouped_matmul(**inputs)
