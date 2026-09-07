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

import logging

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE

_SUPPORTED_VENDORS = {"ascend", "hygon", "iluvatar", "metax", "mthreads", "nvidia"}

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name not in _SUPPORTED_VENDORS,
    reason="int_mm is currently enabled only on the validated accelerator backends",
)

INT_MM_SHAPES = (
    [(3, 7, 5), (3, 5, 0)]
    if QUICK_MODE
    else [
        (0, 5, 3),
        (3, 0, 5),
        (3, 5, 0),
        (1, 1, 1),
        (1, 17, 3),
        (3, 7, 5),
        (16, 24, 32),
        (17, 31, 33),
        (32, 32, 64),
        (65, 67, 33),
        (65, 64, 2048),
    ]
)

INT_MM_LAYOUTS = (
    ["contiguous"]
    if QUICK_MODE
    else [
        "contiguous",
        "transposed",
        "sliced",
        "expanded",
    ]
)


def _make_inputs(M, N, K, layout):
    device = flag_gems.device
    if layout == "contiguous":
        mat1 = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
        mat2 = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=device)
    elif layout == "transposed":
        mat1 = torch.randint(-128, 128, (K, M), dtype=torch.int8, device=device).t()
        mat2 = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=device).t()
    elif layout == "sliced":
        mat1 = torch.randint(-128, 128, (M, K * 2), dtype=torch.int8, device=device)[
            :, ::2
        ]
        mat2 = torch.randint(
            -128, 128, (K * 2, N * 2), dtype=torch.int8, device=device
        )[::2, ::2]
    else:
        mat1 = torch.randint(-128, 128, (1, K), dtype=torch.int8, device=device)
        mat2 = torch.randint(-128, 128, (K, 1), dtype=torch.int8, device=device)
        mat1 = mat1.expand(M, K)
        mat2 = mat2.expand(K, N)
    return mat1, mat2


def _reference_int_mm(mat1, mat2):
    ref1 = mat1.to(device="cpu", dtype=torch.int64)
    ref2 = mat2.to(device="cpu", dtype=torch.int64)
    return (ref1 @ ref2).to(torch.int32)


def _full_on_device(shape, fill_value, dtype, device=None):
    target_device = device or flag_gems.device
    value = torch.full(shape, fill_value, dtype=dtype, device="cpu")
    return value if target_device == "cpu" else value.to(target_device)


def _assert_gems_route(caplog, use_out):
    M, N, K = 3, 7, 5
    mat1, mat2 = _make_inputs(M, N, K, "contiguous")
    op_name = "int_mm_out" if use_out else "int_mm"
    implementation = getattr(flag_gems, op_name)
    logger_name = implementation.__module__
    implementation_logger = logging.getLogger(logger_name)

    implementation_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            if use_out:
                out = torch.empty((M, N), dtype=torch.int32, device=flag_gems.device)
                flag_gems.int_mm_out(mat1, mat2, out=out)
            else:
                flag_gems.int_mm(mat1, mat2)
    finally:
        implementation_logger.removeHandler(caplog.handler)

    expected_suffix = "INT_MM_OUT" if use_out else "INT_MM"
    assert any(
        record.name == logger_name and record.getMessage().endswith(expected_suffix)
        for record in caplog.records
    ), f"{op_name} did not call the selected FlagGems implementation"


@pytest.mark.int_mm
def test_int_mm_gems_route(caplog):
    _assert_gems_route(caplog, use_out=False)


@pytest.mark.int_mm_out
def test_int_mm_out_gems_route(caplog):
    _assert_gems_route(caplog, use_out=True)


@pytest.mark.int_mm
@pytest.mark.parametrize("M, N, K", INT_MM_SHAPES)
@pytest.mark.parametrize("layout", INT_MM_LAYOUTS)
def test_int_mm(M, N, K, layout):
    mat1, mat2 = _make_inputs(M, N, K, layout)
    reference = _reference_int_mm(mat1, mat2)

    result = flag_gems.int_mm(mat1, mat2)

    assert result.dtype == torch.int32
    assert result.shape == (M, N)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.int_mm_out
@pytest.mark.parametrize("M, N, K", INT_MM_SHAPES)
@pytest.mark.parametrize("layout", INT_MM_LAYOUTS)
def test_int_mm_out(M, N, K, layout):
    mat1, mat2 = _make_inputs(M, N, K, layout)
    reference = _reference_int_mm(mat1, mat2)
    out = _full_on_device((M, N), -1103515245, torch.int32)

    result = flag_gems.int_mm_out(mat1, mat2, out=out)

    assert result is out
    assert result.dtype == torch.int32
    assert result.shape == (M, N)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.int_mm
def test_int_mm_int32_overflow():
    K = 131073
    mat1 = _full_on_device((1, K), -128, torch.int8)
    mat2 = _full_on_device((K, 1), -128, torch.int8)
    reference = _reference_int_mm(mat1, mat2)

    result = flag_gems.int_mm(mat1, mat2)

    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.int_mm_out
def test_int_mm_out_int32_overflow():
    K = 131073
    mat1 = _full_on_device((1, K), -128, torch.int8)
    mat2 = _full_on_device((K, 1), -128, torch.int8)
    reference = _reference_int_mm(mat1, mat2)
    out = _full_on_device((1, 1), -1103515245, torch.int32)

    result = flag_gems.int_mm_out(mat1, mat2, out=out)

    assert result is out
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.int_mm
def test_int_mm_observes_mat2_inplace_updates():
    M, N, K = 32, 64, 128
    mat1 = _full_on_device((M, K), 1, torch.int8)
    mat2 = _full_on_device((K, N), 1, torch.int8)

    first = flag_gems.int_mm(mat1, mat2)
    mat2.fill_(2)
    second = flag_gems.int_mm(mat1, mat2)

    torch.testing.assert_close(
        first.cpu(), torch.full((M, N), K, dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        second.cpu(), torch.full((M, N), 2 * K, dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.int_mm
def test_int_mm_observes_inference_mat2_inplace_updates():
    M, N, K = 32, 64, 128
    with torch.inference_mode():
        mat1 = _full_on_device((M, K), 1, torch.int8)
        mat2 = _full_on_device((K, N), 1, torch.int8)
        first = flag_gems.int_mm(mat1, mat2)
        mat2.fill_(2)
        second = flag_gems.int_mm(mat1, mat2)

    torch.testing.assert_close(
        first.cpu(), torch.full((M, N), K, dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        second.cpu(), torch.full((M, N), 2 * K, dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.int_mm
def test_int_mm_does_not_reuse_packed_mat2_for_distinct_alias():
    M, N, K = 32, 64, 128
    mat1 = _full_on_device((M, K), 1, torch.int8)
    storage = torch.randint(
        -128,
        128,
        (K * N,),
        dtype=torch.int8,
        device=flag_gems.device,
    )
    row_major = storage.as_strided((K, N), (N, 1))
    column_major = storage.as_strided((K, N), (1, K))
    assert row_major is not column_major
    assert row_major.data_ptr() == column_major.data_ptr()
    row_major_reference = _reference_int_mm(mat1, row_major)
    column_major_reference = _reference_int_mm(mat1, column_major)
    assert not torch.equal(row_major_reference, column_major_reference)

    row_major_result = flag_gems.int_mm(mat1, row_major)
    column_major_result = flag_gems.int_mm(mat1, column_major)

    torch.testing.assert_close(
        row_major_result.cpu(), row_major_reference, rtol=0, atol=0
    )
    torch.testing.assert_close(
        column_major_result.cpu(), column_major_reference, rtol=0, atol=0
    )


@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon",
    reason="Regression for Hygon's packed-weight cache and HIP graph support",
)
@pytest.mark.parametrize(
    "use_out",
    [
        pytest.param(False, id="default", marks=pytest.mark.int_mm),
        pytest.param(True, id="out", marks=pytest.mark.int_mm_out),
    ],
)
def test_int_mm_graph_replay_observes_weight_updates(use_out):
    M, N, K = 32, 256, 256
    mat1 = _full_on_device((M, K), 1, torch.int8)
    mat2 = _full_on_device((K, N), 1, torch.int8)
    out = torch.empty((M, N), dtype=torch.int32, device=flag_gems.device)
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    def run():
        if use_out:
            return flag_gems.int_mm_out(mat1, mat2, out=out)
        return flag_gems.int_mm(mat1, mat2)

    torch.cuda.synchronize()
    # Warm the cache on the capture stream itself, so a stream-keyed cache
    # cannot accidentally hide a missing pack operation in the graph.
    with torch.cuda.stream(capture_stream):
        run()
    capture_stream.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        result = run()
    torch.cuda.synchronize()
    mat2.fill_(3)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    reference = torch.full((M, N), 3 * K, dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon",
    reason="Regression for Hygon's asynchronous packed-weight cache",
)
@pytest.mark.parametrize(
    "use_out",
    [
        pytest.param(False, id="default", marks=pytest.mark.int_mm),
        pytest.param(True, id="out", marks=pytest.mark.int_mm_out),
    ],
)
def test_int_mm_pack_ready_on_consuming_stream(use_out):
    M, N, K = 32, 256, 256
    mat1 = _full_on_device((M, K), 1, torch.int8)
    mat2 = _full_on_device((K, N), 1, torch.int8)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer_done = torch.cuda.Event()
    out1 = torch.empty((M, N), dtype=torch.int32, device=flag_gems.device)
    out2 = torch.empty_like(out1)

    def run(out):
        if use_out:
            return flag_gems.int_mm_out(mat1, mat2, out=out)
        return flag_gems.int_mm(mat1, mat2)

    # The original inputs are ready; only the internal pack is delayed.
    torch.cuda.synchronize()
    for stream, out in ((producer, out1), (consumer, out2)):
        with torch.cuda.stream(stream):
            run(out)
    torch.cuda.synchronize()
    mat2.fill_(3)
    torch.cuda.synchronize()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(200_000_000)
        first = run(out1)
        producer_done.record()
    with torch.cuda.stream(consumer):
        second = run(out2)
    consumer.synchronize()
    overlapped = not producer_done.query()
    observed = second.cpu()
    torch.cuda.synchronize()

    assert overlapped, "controlled stream overlap was not established"
    reference = torch.full((M, N), 3 * K, dtype=torch.int32)
    torch.testing.assert_close(first.cpu(), reference, rtol=0, atol=0)
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.int_mm_out
def test_int_mm_out_noncontiguous():
    M, N, K = 17, 31, 33
    mat1, mat2 = _make_inputs(M, N, K, "sliced")
    reference = _reference_int_mm(mat1, mat2)
    out = _full_on_device((N, M), -1103515245, torch.int32).t()
    expected_stride = out.stride()

    result = flag_gems.int_mm_out(mat1, mat2, out=out)

    assert result is out
    assert result.stride() == expected_stride
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


@pytest.mark.int_mm_out
def test_int_mm_out_k_zero():
    M, N, K = 3, 5, 0
    mat1, mat2 = _make_inputs(M, N, K, "contiguous")
    out = _full_on_device((M, N), -1103515245, torch.int32)

    result = flag_gems.int_mm_out(mat1, mat2, out=out)

    assert result is out
    torch.testing.assert_close(result.cpu(), torch.zeros((M, N), dtype=torch.int32))


@pytest.mark.int_mm
@pytest.mark.parametrize(
    "mat1_shape, mat2_shape, mat1_dtype, mat2_dtype, match",
    [
        ((3,), (3, 5), torch.int8, torch.int8, "self must be a matrix"),
        ((1, 2, 3), (3, 5), torch.int8, torch.int8, "self must be a matrix"),
        ((3, 5), (5,), torch.int8, torch.int8, "mat2 must be a matrix"),
        ((3, 5), (1, 5, 7), torch.int8, torch.int8, "mat2 must be a matrix"),
        ((3, 5), (5, 7), torch.float32, torch.int8, "expected both inputs"),
        ((3, 5), (5, 7), torch.int8, torch.int32, "expected both inputs"),
        ((3, 5), (6, 7), torch.int8, torch.int8, "cannot be multiplied"),
    ],
)
def test_int_mm_invalid_inputs(mat1_shape, mat2_shape, mat1_dtype, mat2_dtype, match):
    mat1 = torch.empty(mat1_shape, dtype=mat1_dtype, device=flag_gems.device)
    mat2 = torch.empty(mat2_shape, dtype=mat2_dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError, match=match):
        flag_gems.int_mm(mat1, mat2)


@pytest.mark.int_mm
def test_int_mm_mixed_input_devices():
    mat1 = torch.empty((3, 5), dtype=torch.int8, device=flag_gems.device)
    mat2 = torch.empty((5, 7), dtype=torch.int8, device="cpu")

    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.int_mm(mat1, mat2)


@pytest.mark.int_mm_out
@pytest.mark.parametrize(
    "out_shape, out_dtype, out_device, match",
    [
        ((3, 7), torch.float32, None, "dtype torch.int32"),
        ((7, 3), torch.int32, None, "expected out shape"),
        ((3, 7), torch.int32, "cpu", "same device"),
    ],
)
def test_int_mm_out_validation(out_shape, out_dtype, out_device, match):
    mat1 = torch.empty((3, 5), dtype=torch.int8, device=flag_gems.device)
    mat2 = torch.empty((5, 7), dtype=torch.int8, device=flag_gems.device)
    out = _full_on_device(
        out_shape,
        -1103515245,
        out_dtype,
        device=out_device,
    )
    before = out.clone()

    with pytest.raises(RuntimeError, match=match):
        flag_gems.int_mm_out(mat1, mat2, out=out)

    torch.testing.assert_close(out, before, rtol=0, atol=0)
