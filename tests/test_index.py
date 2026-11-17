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

import random
import time

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

INDEX_ACC_SHAPE = (
    # Keep these cases aligned with benchmark/test_index.py so benchmark
    # latency rows have matching correctness coverage.
    ((2**28,), ((2**16,),)),
    ((32, 32), ((8,), (8,))),
    ((32, 32), ((8,), (2, 8))),
    ((32, 32), ((2, 8),)),
    ((1024, 1024), ((64,), (64,))),
    ((512, 512, 512), ((128,), (128,), (128,))),
    ((512, 512, 512), ((2, 128), (2, 128), (2, 128))),
    ((512, 512, 512), ((2, 128), (128,), (128,))),
    ((512, 512, 512), ((2, 128),)),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
    ),
)

INDEX_NONLEADING_ADJACENT_SHAPE = (
    ((1, 4096, 512), (None, (32768,), None)),
    ((4, 512, 128), (None, (4096,), None)),
    ((2, 256, 256, 64), (None, (4096,), (4096,), None)),
    ((2, 128, 128, 128), (None, (2048,), (2048,), (2048,))),
    (
        (1, 128, 128, 64, 8),
        (None, (2048,), (2048,), (2048,), None),
    ),
)

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def gen_indices(input_shape, indices_shape, accumulate):
    """
    Generate indices for torch.ops.aten.index while preserving benchmark shapes.
    """
    indices = []
    for dim, shape in enumerate(indices_shape):
        index = np.random.choice(
            np.arange(input_shape[dim]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


def gen_optional_indices(input_shape, indices_shape, accumulate):
    indices = []
    for dim, shape in enumerate(indices_shape):
        if shape is None:
            indices.append(None)
            continue
        index = np.random.choice(
            np.arange(input_shape[dim]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


@pytest.mark.index
@pytest.mark.parametrize("input_shape, indices_shape", INDEX_ACC_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index(input_shape, indices_shape, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    try:
        indices = gen_indices(input_shape, indices_shape, True)
    except Exception:
        return False

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    try:
        ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    except (IndexError, RuntimeError):
        return False

    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize(
    "input_shape, indices_shape",
    INDEX_NONLEADING_ADJACENT_SHAPE,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_nonleading_adjacent_tensor_indices(input_shape, indices_shape, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_optional_indices(input_shape, indices_shape, True)

    ref_inp = utils.to_reference(inp)
    ref_indices = [
        None if index is None else utils.to_reference(index) for index in indices
    ]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


# Additional test cases to improve coverage for index operator
@pytest.mark.index
@pytest.mark.parametrize(
    "input_shape, index_pos",
    [
        ((32, 32), 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_with_none_basic_indexing(input_shape, index_pos, dtype):
    """Test basic indexing with None (ellipsis-like behavior)"""
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    indices = [None] * len(input_shape)

    # Add a single tensor index at the specified position
    idx = torch.randint(0, input_shape[index_pos], (8,), device=flag_gems.device)
    indices[index_pos] = idx

    ref_inp = utils.to_reference(inp)
    ref_indices = [None if idx is None else utils.to_reference(idx) for idx in indices]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize(
    "input_shape, indices_idx",
    # 0 in indices_idx means a Tensor
    # 1 in indices_idx means None
    [
        ((1024, 1024), (0, 1)),
        ((16, 16, 16), (1, 0, 0)),
        ((16, 16, 16), (0, 1, 0)),
        ((32, 32, 32), (0, 0, 1)),
        ((32, 32, 32), (1, 1, 0)),
        ((64, 64, 64), (1, 0, 1)),
        ((64, 64, 64), (0, 1, 1)),
        ((12, 12, 12, 12), (1, 0, 0, 0)),
        ((12, 12, 12, 12), (0, 1, 0, 0)),
        ((10, 10, 10, 10), (0, 0, 1, 0)),
        ((10, 10, 10, 10), (0, 0, 0, 1)),
        ((10, 10, 10, 10), (1, 1, 0, 0)),
        ((10, 10, 10, 10), (1, 0, 1, 0)),
        ((16, 16, 16, 16), (1, 0, 0, 1)),
        ((16, 16, 16, 16), (0, 1, 1, 0)),
        ((32, 32, 32, 32), (0, 1, 0, 1)),
        ((32, 32, 32, 32), (0, 0, 1, 1)),
        ((8, 8, 8, 8), (0, 1, 1, 1)),
        ((8, 8, 8, 8), (1, 0, 1, 1)),
        ((8, 8, 8, 8), (1, 1, 0, 1)),
        ((8, 8, 8, 8), (1, 1, 1, 0)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.int64])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_with_none_and_tensor(input_shape, indices_idx, dtype):
    inp = torch.randint(0, 10000, input_shape, dtype=dtype, device=flag_gems.device)
    indices = []
    random_idx_list_len = random.randint(0, min(input_shape) - 1)
    for i, idx_pos in enumerate(indices_idx):
        if idx_pos:
            indices.append(None)
        else:
            dim_len = input_shape[i]
            random_idx = random.randint(0, dim_len - 1)
            indices.append(
                torch.tensor(
                    [random_idx for _ in range(random_idx_list_len)],
                    device=flag_gems.device,
                    dtype=dtype,
                )
            )

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(x) for x in indices]
    result_ref_ = torch.ops.aten.index(ref_inp, ref_indices)
    result_gems_ = flag_gems.index(inp, indices)

    utils.gems_assert_close(result_gems_, result_ref_, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_boolean_mask(dtype):
    """Test boolean mask indexing"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    mask = torch.rand(32, 64, device=flag_gems.device) > 0.5
    indices = [mask]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(mask)]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_empty_tensor(dtype):
    """Test index with empty tensor"""

    inp = torch.empty((0, 32), dtype=dtype, device=flag_gems.device)
    idx = torch.empty((0,), dtype=torch.long, device=flag_gems.device)
    indices = [idx, None]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx), None]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_1d_special_case(dtype):
    """Test 1D input special case (uses gather)"""

    inp = torch.randn((128,), dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, 128, (16,), device=flag_gems.device)
    indices = [idx]

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx)]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_error_empty_indices(dtype):
    """Test error handling: empty indices"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = []

    with pytest.raises(ValueError, match="at least one index must be provided"):
        flag_gems.index(inp, indices)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_error_too_many_indices(dtype):
    """Test error handling: too many indices"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    idx1 = torch.randint(0, 32, (8,), device=flag_gems.device)
    idx2 = torch.randint(0, 64, (8,), device=flag_gems.device)
    idx3 = torch.randint(0, 32, (8,), device=flag_gems.device)
    indices = [idx1, idx2, idx3]  # Too many for 2D tensor

    with pytest.raises(IndexError, match="too many indices"):
        flag_gems.index(inp, indices)


@pytest.mark.index
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_int32_offset_overflow():
    """Regression for issue #4153: int64 offsets for large element ranges.

    A (2**28 + 1, 8) int8 tensor holds 2**31 + 8 elements, so the int32
    index value 2**28 produces element offsets >= 2**31. These used to
    overflow int32 arithmetic in the generated kernels and caused
    out-of-bounds reads ("illegal memory access").
    """
    free_bytes, _ = torch.cuda.mem_get_info(flag_gems.device)
    if free_bytes < 3 * 2**30:
        pytest.skip("not enough free device memory for the int32-offset case")

    rows = 2**28 + 1
    inp = torch.zeros((rows, 8), dtype=torch.int8, device=flag_gems.device)
    idx = torch.tensor([2**28], dtype=torch.int32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_idx = utils.to_reference(idx)

    # Contiguous input takes the linearized adjacent-index kernel.
    ref_out = torch.ops.aten.index(ref_inp, [ref_idx])
    out = flag_gems.index(inp, [idx])
    utils.gems_assert_close(out, ref_out, torch.int8)

    # Transposed view takes the generic (code-generated) kernel.
    ref_out_t = torch.ops.aten.index(ref_inp.t(), [None, ref_idx])
    out_t = flag_gems.index(inp.t(), [None, idx])
    utils.gems_assert_close(out_t, ref_out_t, torch.int8)


@pytest.mark.index
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("use_graph", [False, True])
def test_index_large_strided_int32_offset(index_dtype, use_graph):
    """Widen loaded indices before multiplying a large state-cache stride."""
    import gc

    if flag_gems.device != "cuda":
        pytest.skip("requires a CUDA-compatible device for memory and graph APIs")
    shape = (9109, 16, 128, 128)
    strides = (294912, 16384, 128, 1)
    slots = [7280, 7281, 7282, 7296, 9108]
    guard = 2**31
    row_size = 16 * 128 * 128
    elements = guard + (shape[0] - 1) * strides[0] + row_size
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < elements * 2 + 1024**3:
        pytest.skip("requires memory for a guarded BF16 large-offset regression")

    backing = state = indices = packed_indices = actual = expected = graph = None
    try:
        # Old int32 negative offsets stay inside this allocation's prefix.
        backing = torch.empty(elements, dtype=torch.bfloat16, device=flag_gems.device)
        backing[:guard].fill_(-7)
        state = torch.as_strided(backing, shape, strides, storage_offset=guard)
        for position, slot in enumerate(slots):
            state[slot].fill_(11 + position)
        packed_indices = torch.tensor(
            [value for slot in slots for value in (slot, 0)],
            dtype=index_dtype,
            device=flag_gems.device,
        )
        indices = packed_indices[::2]
        expected = torch.stack([state[slot] for slot in slots])

        def invoke():
            return flag_gems.index(state, [indices])

        if use_graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    actual = invoke()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = invoke()
            for delta in (20, 40):
                for position, slot in enumerate(slots):
                    state[slot].fill_(11 + position + delta)
                    expected[position].fill_(11 + position + delta)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            actual = invoke()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        graph = actual = expected = state = backing = indices = packed_indices = None
        gc.collect()
        torch.cuda.empty_cache()


def _index_strided_view(raw, layout):
    if layout == "transpose":
        return raw.transpose(0, 1)
    if layout == "offset_permute":
        return raw[1:5].permute(2, 0, 1)
    return raw[1:5:2, 1:6:2, 1:8:2]


@pytest.mark.index
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32]
)
@pytest.mark.parametrize("layout", ["transpose", "offset_permute", "slice"])
@pytest.mark.parametrize("mode", ["single", "paired", "broadcast"])
def test_index_noncontiguous_leading(dtype, layout, mode):
    cpu_raw = (torch.arange(6 * 7 * 8) % 97).reshape(6, 7, 8).to(dtype)
    raw = cpu_raw.to(flag_gems.device)
    inp = _index_strided_view(raw, layout)
    cpu = _index_strided_view(cpu_raw, layout)
    cpu_indices = [torch.tensor([0, inp.size(0) - 1])]
    if mode != "single":
        cpu_indices.append(torch.tensor([0, inp.size(1) - 1]))
    if mode == "broadcast":
        cpu_indices = [cpu_indices[0][:, None], cpu_indices[1][None, :]]
    expected = torch.ops.aten.index.Tensor(cpu, cpu_indices)
    indices = [item.to(inp.device) for item in cpu_indices]
    actual = flag_gems.index(inp, indices)
    utils.gems_assert_equal(actual.cpu(), expected)
    utils.gems_assert_equal(raw.cpu(), cpu_raw)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["transpose", "slice"])
def test_index_noncontiguous_negative(dtype, layout):
    cpu_raw = (torch.arange(6 * 7 * 8) % 97).reshape(6, 7, 8).to(dtype)
    raw = cpu_raw.to(flag_gems.device)
    cpu = _index_strided_view(cpu_raw, layout)
    cpu_indices = [torch.tensor([-1, 0]), torch.tensor([-1, 0])]
    actual = flag_gems.index(
        _index_strided_view(raw, layout), [i.to(raw.device) for i in cpu_indices]
    )
    utils.gems_assert_equal(actual.cpu(), torch.ops.aten.index.Tensor(cpu, cpu_indices))


@pytest.mark.index
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Requires Ascend NPUGraph capture and replay",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["transpose", "slice"])
def test_index_noncontiguous_graph(dtype, layout):
    cpu_raw = (torch.arange(6 * 7 * 8) % 97).reshape(6, 7, 8).to(dtype)
    raw = cpu_raw.to(flag_gems.device)
    inp = _index_strided_view(raw, layout)
    index = torch.tensor([0, inp.size(0) - 1], device=inp.device)
    flag_gems.index(inp, [index])
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = flag_gems.index(inp, [index])
    for updated, positions in ((-cpu_raw, [inp.size(0) - 1, 0]), (cpu_raw + 2, [0, 1])):
        raw.copy_(updated)
        index.copy_(torch.tensor(positions))
        actual.fill_(-123)
        graph.replay()
        torch.npu.synchronize()
        expected = torch.ops.aten.index.Tensor(
            _index_strided_view(updated, layout), [torch.tensor(positions)]
        )
        utils.gems_assert_equal(actual.cpu(), expected)
        utils.gems_assert_equal(raw.cpu(), updated)


@pytest.mark.index
@pytest.mark.parametrize("payload_dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize("pattern", ["none", "all", "strided", "multiblock"])
def test_index_mask_coordinate_order(payload_dtype, mask_dtype, pattern):
    shape = (65, 129, 2) if pattern == "multiblock" else (3, 5, 4)
    cpu = torch.arange(np.prod(shape), dtype=torch.int64).reshape(shape)
    if payload_dtype == torch.int64:
        cpu = cpu + 2**54
    cpu = cpu.to(payload_dtype)
    mask_cpu = (torch.arange(shape[0] * shape[1]).reshape(shape[:2]) % 3 == 0).to(
        mask_dtype
    )
    if pattern == "none":
        mask_cpu.zero_()
    elif pattern == "all":
        mask_cpu.fill_(1)
    inp = cpu.to(flag_gems.device)
    if pattern == "strided":
        storage = torch.empty((shape[1], shape[0]), dtype=mask_dtype, device=inp.device)
        mask = storage.t()
        mask.copy_(mask_cpu)
    else:
        mask = mask_cpu.to(inp.device)
    actual = flag_gems.index(inp, [mask])
    expected = torch.ops.aten.index.Tensor(cpu, [mask_cpu])
    utils.gems_assert_equal(actual.cpu(), expected)
    utils.gems_assert_equal(inp.cpu(), cpu)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("mode", ["scalar", "nonleading", "nonadjacent", "broadcast"])
def test_index_tensor_metadata_forms(dtype, mode):
    cpu = torch.arange(120, dtype=torch.int64).reshape(4, 5, 6)
    if dtype == torch.int64:
        cpu = cpu + 2**54
    cpu = cpu.to(dtype)
    if mode == "scalar":
        indices = [torch.tensor(-1), None, torch.tensor(-1)]
    elif mode == "nonleading":
        indices = [None, torch.tensor([0, -1])]
    elif mode == "nonadjacent":
        indices = [torch.tensor([0, -1]), None, torch.tensor([-1, 0])]
    else:
        indices = [torch.tensor([[0], [-1]]), torch.tensor([[0, 2, -1]])]
    device_indices = [
        item.to(flag_gems.device) if item is not None else None for item in indices
    ]
    actual = flag_gems.index(cpu.to(flag_gems.device), device_indices)
    utils.gems_assert_equal(actual.cpu(), torch.ops.aten.index.Tensor(cpu, indices))


@pytest.mark.index
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend data-dependent graph output boundary",
)
def test_index_boolean_capture_rejected():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    mask = torch.tensor([True, False, True], device=inp.device)
    scratch = torch.empty_like(inp)
    graph = torch.npu.NPUGraph()
    with pytest.raises(RuntimeError, match="data-dependent output shape"):
        with torch.npu.graph(graph):
            scratch.copy_(inp)
            flag_gems.index(inp, [mask])
    utils.gems_assert_equal(
        inp.cpu(), torch.arange(12, dtype=torch.float32).reshape(3, 4)
    )


@pytest.mark.index
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend eager bounds validation"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "separated"])
@pytest.mark.parametrize("bad", [4, -5, 2**63 - 1, -(2**63)])
def test_index_out_of_bounds_rejected(dtype, layout, bad):
    cpu = torch.arange(4 * 4 * 6).reshape(4, 4, 6).to(dtype)
    if layout == "transpose":
        cpu = cpu.transpose(0, 1)
    indices = [torch.tensor([0, bad])]
    if layout == "separated":
        indices += [None, torch.tensor([-1, 0])]
    with pytest.raises(IndexError):
        torch.ops.aten.index.Tensor(cpu, indices)
    inp = cpu.to(flag_gems.device)
    device_indices = [
        item.to(inp.device) if item is not None else None for item in indices
    ]
    with pytest.raises(IndexError, match="out of bounds"):
        flag_gems.index(inp, device_indices)
    utils.gems_assert_equal(inp.cpu(), cpu)


@pytest.mark.index
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend eager bounds validation"
)
@pytest.mark.parametrize("count", [1025, 65537])
@pytest.mark.parametrize("bad", [4, -5])
def test_index_out_of_bounds_block_tail(count, bad):
    inp = torch.arange(24, device=flag_gems.device).reshape(4, 6)
    cpu_storage = torch.zeros(2 * count, dtype=torch.int64)
    cpu_storage[-2] = bad
    index = cpu_storage.to(inp.device)[::2]
    with pytest.raises(IndexError, match="out of bounds"):
        flag_gems.index(inp, [index])
    index[-1] = -4
    actual = flag_gems.index(inp, [index])
    expected = torch.zeros((count, 6), dtype=torch.int64)
    expected[:] = torch.arange(6)
    utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("layout", ["transpose", "slice"])
@pytest.mark.parametrize("paired", [False, True])
def test_index_noncontiguous_suffix_tail(dtype, layout, paired):
    cpu_raw = torch.arange(7 * 3 * 1301, dtype=torch.int64).reshape(7, 3, 1301)
    cpu_raw = cpu_raw + 2**54 if dtype == torch.int64 else cpu_raw % 97
    cpu_raw = cpu_raw.to(dtype)
    raw = cpu_raw.to(flag_gems.device)
    if layout == "transpose":
        cpu, inp = cpu_raw.transpose(0, 1), raw.transpose(0, 1)
    else:
        cpu, inp = cpu_raw[::2, :, ::2], raw[::2, :, ::2]
    indices = [torch.tensor([0, -1])]
    if paired:
        indices.append(torch.tensor([-1, 0]))
    expected = torch.ops.aten.index.Tensor(cpu, indices)
    actual = flag_gems.index(inp, [index.to(inp.device) for index in indices])
    utils.gems_assert_equal(actual.cpu(), expected)
    utils.gems_assert_equal(raw.cpu(), cpu_raw)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("extent", [1025, 4097])
def test_index_large_sparse_noncontiguous(dtype, extent):
    values = (
        torch.arange(17, dtype=torch.int16)[:, None, None]
        + torch.arange(extent, dtype=torch.int16)[None, :, None]
        + torch.arange(1025, dtype=torch.int16)[None, None, :]
    )
    cpu = (values % 97).to(dtype)
    inp = cpu.to(flag_gems.device).transpose(0, 1)
    index = torch.tensor([0, -1], device=inp.device)
    actual = flag_gems.index(inp, [index])
    expected = cpu.transpose(0, 1)[torch.tensor([0, -1])]
    utils.gems_assert_equal(actual.cpu(), expected)
