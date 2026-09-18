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

FP8_DTYPE = torch.float8_e5m2
FP8_GROUP_SIZE = 128


def _fp8_available():
    if flag_gems.device == "musa":
        return torch.musa.is_available() and hasattr(torch, "float8_e4m3fn")
    return (
        torch.cuda.is_available()
        and hasattr(torch, "float8_e5m2")
        and (
            flag_gems.vendor_name != "nvidia"
            or torch.cuda.get_device_capability()[0] >= 9
        )
    )


def _quantize_fp8_grouped(x, group_size=FP8_GROUP_SIZE, fp8_dtype=FP8_DTYPE):
    fp8_info = torch.finfo(fp8_dtype)
    *leading, n = x.shape
    padded = (n + group_size - 1) // group_size * group_size
    x_pad = torch.nn.functional.pad(x.float(), (0, padded - n))
    grouped = x_pad.reshape(*leading, padded // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
    q = (grouped / scale).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
    return (
        q.reshape(*leading, padded)[..., :n].contiguous(),
        scale.squeeze(-1).to(x.dtype).contiguous(),
    )


def _dequant_fp8(x_fp8, x_scale, group_size=FP8_GROUP_SIZE):
    *leading, n = x_fp8.shape
    num_groups = x_scale.shape[-1]
    padded = num_groups * group_size
    x_pad = torch.nn.functional.pad(x_fp8.float(), (0, padded - n))
    grouped = x_pad.reshape(*leading, num_groups, group_size)
    dequant = grouped * x_scale.unsqueeze(-1).float()
    return dequant.reshape(*leading, padded)[..., :n].to(x_scale.dtype)


@pytest.mark.topk_w8a16_fp8
@pytest.mark.skipif(
    getattr(flag_gems, "vendor_name", None)
    not in ("thead", "hygon", "mthreads", "nvidia"),
    reason="topk_w8a16_fp8 requires an implemented backend",
)
@pytest.mark.skipif(not _fp8_available(), reason="required FP8 format is unavailable")
@pytest.mark.parametrize(
    "shape, k",
    [
        ((4, 128), 5),
        ((8, 256), 8),
        ((4, 1024), 16),
        ((2, 4096), 32),
        ((2, 8192), 64),
        ((8, 32768), 256),
        ((2, 33, 128), 8),
    ],
)
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize(
    "fp8_dtype",
    [
        pytest.param(
            torch.float8_e5m2,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name in ("hygon", "mthreads", "nvidia"),
                reason="Hygon, MThreads and NVIDIA use E4M3FN coverage",
            ),
        ),
        pytest.param(
            torch.float8_e4m3fn,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name not in ("hygon", "mthreads", "nvidia"),
                reason="E4M3FN extension requires Hygon, MThreads or NVIDIA",
            ),
        ),
    ],
)
def test_topk_w8a16_fp8(shape, k, largest, fp8_dtype):
    dtype = torch.bfloat16
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_fp8, x_scale = _quantize_fp8_grouped(x, fp8_dtype=fp8_dtype)
    dequant = _dequant_fp8(x_fp8, x_scale)

    ref_value, ref_index = torch.topk(
        utils.to_reference(dequant), k, dim=-1, largest=largest, sorted=True
    )
    res_value, res_index = flag_gems.topk_w8a16_fp8(
        x_fp8,
        x_scale,
        k,
        dim=-1,
        largest=largest,
        sorted=True,
        group_size=FP8_GROUP_SIZE,
        out_dtype=dtype,
    )

    utils.gems_assert_close(res_value, ref_value, dtype)
    # FP8 E5M2 quantization creates many ties; index order among equal
    # values may differ from torch.topk. Check the selected values instead.
    utils.gems_assert_close(torch.gather(dequant, -1, res_index), ref_value, dtype)


NVIDIA_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or not _fp8_available(),
    reason="NVIDIA FP8 TopK requires compute capability >= 9.0",
)


@NVIDIA_ONLY
@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("shape, topk", [((4, 128), 5), ((8, 256), 16), ((2, 1024), 8)])
@pytest.mark.parametrize("largest", [True, False])
def test_topk_w8a16_fp8_grouped_scale_nvidia(shape, topk, largest):
    x = torch.randn(shape, dtype=torch.bfloat16, device=flag_gems.device)
    x_fp8, x_scale = _quantize_fp8_grouped(x, fp8_dtype=torch.float8_e4m3fn)
    group_ids = torch.arange(shape[-1], device=x.device) // FP8_GROUP_SIZE
    x_dequant = x_fp8.float() * x_scale.index_select(-1, group_ids).float()

    ref_value = torch.topk(x_dequant, topk, dim=-1, largest=largest, sorted=True).values
    res_value, res_index = flag_gems.topk_w8a16_fp8(
        x_fp8, x_scale, topk, dim=-1, largest=largest, sorted=True
    )

    gathered = torch.gather(x_dequant, dim=-1, index=res_index)
    torch.testing.assert_close(res_value.float(), ref_value, rtol=0, atol=2e-2)
    torch.testing.assert_close(gathered, res_value.float(), rtol=0, atol=2e-2)


@NVIDIA_ONLY
@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("shape, topk", [((4, 128), 5), ((8, 256), 16), ((2, 4096), 8)])
def test_topk_w8a16_fp8_row_scale(shape, topk):
    x = torch.randn(shape, dtype=torch.bfloat16, device=flag_gems.device)
    x_fp8, x_scale = _quantize_fp8_grouped(
        x, group_size=shape[-1], fp8_dtype=torch.float8_e4m3fn
    )
    x_dequant = x_fp8.float() * x_scale.float()

    ref_value = torch.topk(x_dequant, topk, dim=-1, largest=True, sorted=True).values
    res_value, res_index = flag_gems.topk_w8a16_fp8(
        x_fp8, x_scale, topk, group_size=shape[-1]
    )

    gathered = torch.gather(x_dequant, dim=-1, index=res_index)
    torch.testing.assert_close(res_value.float(), ref_value, rtol=0, atol=2e-2)
    torch.testing.assert_close(gathered, res_value.float(), rtol=0, atol=2e-2)


# E4M3FN extensions stay in the shared operator file. PPU retains its original
# E5M2/BF16 coverage until these additional contracts are supported there.
E4M3_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name not in ("hygon", "mthreads"),
    reason="E4M3FN FP8 extensions require Hygon or Moore Threads",
)


def _device_api():
    return torch.musa if flag_gems.device == "musa" else torch.cuda


def _byte_arange(size):
    return torch.arange(size, dtype=torch.uint8).to(flag_gems.device)


def _cpu_dequant(q, scale, group_size, out_dtype):
    cols = torch.arange(q.shape[-1]) // group_size
    return (q.cpu().float() * scale.cpu().float()[..., cols]).to(out_dtype)


def _check_topk(
    q, scale, k, group_size=128, out_dtype=torch.bfloat16, largest=True, sorted=True
):
    ref = _cpu_dequant(q, scale, group_size, out_dtype)
    values, indices = flag_gems.topk_w8a16_fp8(
        q,
        scale,
        k,
        group_size=group_size,
        out_dtype=out_dtype,
        largest=largest,
        sorted=sorted,
    )
    expected = torch.topk(ref, k, largest=largest, sorted=True).values
    torch.testing.assert_close(values.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert indices.dtype == torch.int64
    torch.testing.assert_close(
        torch.gather(ref, -1, indices.cpu()),
        values.cpu(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    if k > 1:
        assert (indices.cpu().sort().values.diff() != 0).all()
    return values, indices


@E4M3_ONLY
@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "shape,k,group_size",
    [
        ((257,), 257, 33),
        ((3, 259), 17, 64),
        ((2, 2, 1025), 33, 128),
        ((1, 65537), 8, 128),
    ],
)
def test_topk_fp8_edges(shape, k, group_size, largest, out_dtype, fp8_dtype):
    torch.manual_seed(5966)
    q = torch.randn(shape, device=flag_gems.device).to(fp8_dtype)
    scale = torch.randn(
        shape[:-1] + ((shape[-1] + group_size - 1) // group_size,), device=q.device
    )
    _check_topk(q, scale, k, group_size, out_dtype, largest, sorted=False)


@E4M3_ONLY
@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("k", [1, 17, 256])
def test_topk_fp8_all_encodings(fp8_dtype, largest, k):
    q = _byte_arange(256).view(fp8_dtype).reshape(1, 256)
    scale = torch.tensor([[1.0, -2.0]], device=q.device)
    _check_topk(q, scale, k, largest=largest)


@E4M3_ONLY
@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("shape,k", [((0, 128), 3), ((3, 0), 0), ((2, 128), 0)])
def test_topk_fp8_empty(shape, k):
    q = torch.empty(shape, device=flag_gems.device, dtype=torch.float8_e4m3fn)
    s = torch.ones(shape[:-1] + ((shape[-1] + 127) // 128,), device=q.device)
    _check_topk(q, s, k)


@E4M3_ONLY
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_strides_graph_and_ties():
    raw = (
        torch.arange(4 * 514, device=flag_gems.device)
        .remainder(120)
        .byte()
        .reshape(4, 514)
    )
    q = raw[::2, ::2].view(torch.float8_e4m3fn)
    scale = torch.ones((4, 6), device=q.device)[::2, ::2]
    _check_topk(q, scale, 17)
    device_api = _device_api()
    stream = device_api.Stream()
    stream.wait_stream(device_api.current_stream())
    with device_api.stream(stream):
        for _ in range(3):
            flag_gems.topk_w8a16_fp8(q, scale, 17)
    device_api.current_stream().wait_stream(stream)
    graph_type = (
        device_api.MUSAGraph if flag_gems.device == "musa" else device_api.CUDAGraph
    )
    graph = graph_type()
    with device_api.graph(graph):
        v, i = flag_gems.topk_w8a16_fp8(q, scale, 17)
    for change in (0, 1, 2):
        if change == 0:
            scale.neg_()
        if change == 1:
            raw.zero_()
        if change == 2:
            scale.zero_()
        graph.replay()
        ref = _cpu_dequant(q, scale, 128, torch.bfloat16)
        torch.testing.assert_close(v.cpu(), torch.topk(ref, 17).values, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.gather(ref, -1, i.cpu()), v.cpu(), rtol=0, atol=0
        )
        if change > 0:
            torch.testing.assert_close(
                i.cpu(), torch.arange(17).expand(2, 17), rtol=0, atol=0
            )


@E4M3_ONLY
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_validation():
    q = torch.zeros((2, 128), device=flag_gems.device).to(torch.float8_e4m3fn)
    s = torch.ones((2, 1), device=q.device)
    for kwargs in (
        {"k": -1},
        {"k": 129},
        {"k": 1, "dim": 0},
        {"k": 1, "group_size": 0},
    ):
        with pytest.raises(ValueError):
            flag_gems.topk_w8a16_fp8(q, s, **kwargs)
    with pytest.raises(ValueError):
        flag_gems.topk_w8a16_fp8(q, s.expand(2, 2), 1)
    with pytest.raises(TypeError):
        flag_gems.topk_w8a16_fp8(q.float(), s, 1)
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        flag_gems.topk_w8a16_fp8(q.float().to(torch.float8_e5m2), s, 1)
    with pytest.raises(TypeError):
        flag_gems.topk_w8a16_fp8(q, s, 1, out_dtype=torch.float32)
    with pytest.raises(ValueError):
        flag_gems.topk_w8a16_fp8(q, s.cpu(), 1)


@E4M3_ONLY
@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize("n,k", [(4097, 32), (32769, 256)])
@pytest.mark.parametrize("largest", [True, False])
def test_topk_fp8_partition_tails(n, k, largest):
    torch.manual_seed(5966)
    q = torch.randn((2, n), device=flag_gems.device).to(torch.float8_e4m3fn)
    scale = torch.randn((2, (n + 127) // 128), device=q.device, dtype=torch.float16)
    _check_topk(q, scale, k, largest=largest)


@E4M3_ONLY
@pytest.mark.topk_w8a16_fp8
@pytest.mark.parametrize(
    "scale_value",
    [1.0, -1.0, 0.0, float("nan"), float("inf"), 1e-35, 1e35, 0.03125, 100.0],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_topk_fp8_single_group_scales(scale_value, out_dtype):
    q = _byte_arange(256).reshape(2, 128).view(torch.float8_e4m3fn)
    scale = torch.full((2, 1), scale_value, device=q.device)
    for largest in (True, False):
        _check_topk(q, scale, 17, out_dtype=out_dtype, largest=largest)
