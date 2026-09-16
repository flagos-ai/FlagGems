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

from . import accuracy_utils as utils


def _cuda_hopper_w8a8_fp8_available():
    tensor_descriptor = getattr(
        getattr(triton, "tools", None), "tensor_descriptor", None
    )
    return (
        flag_gems.device == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(tensor_descriptor, "TensorDescriptor")
    )


def _mthreads_w8a8_fp8_available():
    return flag_gems.vendor_name == "mthreads" and hasattr(flag_gems, "mm_w8a8_fp8")


def _scaled_mm_backend_available():
    return _cuda_hopper_w8a8_fp8_available() or _mthreads_w8a8_fp8_available()


def _external_scales(a, b, rowwise=False):
    if rowwise:
        sa = torch.linspace(0.25, 1.0, a.shape[0] * 2, device=a.device)[::2]
        sb = torch.linspace(0.5, 1.5, b.shape[1] * 3, device=b.device)[::3]
        return sa, sb
    return (
        torch.full((), 0.5, device=a.device, dtype=torch.float32),
        torch.full((1, 1), 1.5, device=b.device, dtype=torch.float32),
    )


def _scaled_fp8_reference(a, b, sa, sb):
    return (a.float() @ b.float()) * sa.reshape(-1, 1) * sb.reshape(1, -1)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 16, 16),
        (2, 32, 32),
        (8, 64, 64),
        (16, 128, 64),
        (32, 128, 128),
        (64, 256, 128),
        (128, 256, 256),
        (192, 512, 512),
        (256, 768, 1024),
        (512, 1024, 1024),
    ],
)
@pytest.mark.skipif(
    not (_cuda_hopper_w8a8_fp8_available() or _mthreads_w8a8_fp8_available()),
    reason="mm_w8a8_fp8 requires Hopper or MThreads FP8 support",
)
def test_mm_w8a8_fp8(M, N, K):
    dtype = torch.bfloat16
    torch.manual_seed(0)

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    mat1, mat2 = mat1.to(torch.float8_e4m3fn), mat2.to(torch.float8_e4m3fn)
    sa, sb = _external_scales(mat1, mat2)
    args = (mat1, mat2, sa, sb)
    ref_out = utils.to_reference(_scaled_fp8_reference(*args), True)

    res_out = flag_gems.mm_w8a8_fp8(*args, out_dtype=dtype)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    res_out_reused = flag_gems.mm_w8a8_fp8_out(*args, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    utils.gems_assert_close(res_out_reused, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _mthreads_w8a8_fp8_available(), reason="MThreads FP8 backend")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 1, 1),
        (7, 33, 15),
        (16, 128, 64),
        (31, 65, 129),
        (32, 64, 7168),
        (64, 256, 7168),
        (128, 256, 4096),
        (256, 768, 1024),
        (512, 1024, 2048),
        (64, 64, 32768),
        (32, 1024, 8192),
        (128, 1024, 16384),
        (256, 256, 32768),
        (512, 1024, 8192),
        (192, 768, 8193),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("column_major", [True, False])
@pytest.mark.parametrize("rowwise", [False, True])
def test_mm_w8a8_fp8_mthreads_prequantized(M, N, K, dtype, column_major, rowwise):
    torch.manual_seed(42)
    a = torch.randn((M, K), device=flag_gems.device).to(dtype)
    b = torch.randn((N, K) if column_major else (K, N), device=flag_gems.device).to(
        dtype
    )
    if column_major:
        b = b.T
    sa, sb = _external_scales(a, b, rowwise)
    ref = _scaled_fp8_reference(a, b, sa, sb)
    out_storage = torch.empty((M, N * 2), device=a.device, dtype=torch.bfloat16)
    out = out_storage[:, ::2]
    result = flag_gems.mm_w8a8_fp8_out(a, b, sa, sb, out=out)
    assert result is out
    torch.testing.assert_close(result, ref.to(out.dtype), rtol=0.016, atol=0.01)
    result_fp32 = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(result_fp32, ref, rtol=5e-4, atol=0.003)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _mthreads_w8a8_fp8_available(), reason="MThreads FP8 backend")
@pytest.mark.parametrize("M,N,K", [(256, 256, 8192), (128, 1024, 16384)])
@pytest.mark.parametrize("scalar_a", [False, True])
def test_mm_w8a8_fp8_mthreads_large_k_scales(M, N, K, scalar_a):
    torch.manual_seed(42)
    a = torch.randn((M, K), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.randn((N, K), device=flag_gems.device).to(a.dtype).T
    row_a, row_b = _external_scales(a, b, rowwise=True)
    tensor_a, tensor_b = _external_scales(a, b)
    sa = tensor_a if scalar_a else row_a[:, None]
    sb = row_b[None, :] if scalar_a else tensor_b
    ref = _scaled_fp8_reference(a, b, sa, sb)
    result = flag_gems.mm_w8a8_fp8(
        a, b, scale_a=sa, scale_b=sb, out_dtype=torch.float32
    )
    torch.testing.assert_close(result, ref, rtol=5e-4, atol=0.003)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _scaled_mm_backend_available(), reason="FP8 scaled MM backend")
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (256, 256, 8192)])
def test_mm_w8a8_fp8_graph_updates(M, N, K):
    a = torch.randn((M, K), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.randn((N, K), device=flag_gems.device).to(a.dtype).T
    sa, sb = _external_scales(a, b, rowwise=True)
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    backend = torch.cuda if flag_gems.device == "cuda" else torch.musa
    graph_cls = backend.CUDAGraph if flag_gems.device == "cuda" else backend.MUSAGraph
    stream = backend.Stream()
    stream.wait_stream(backend.current_stream())
    with backend.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_fp8_out(a, b, sa, sb, out=out)
        graph = graph_cls()
        with backend.graph(graph):
            flag_gems.mm_w8a8_fp8_out(a, b, sa, sb, out=out)
    backend.current_stream().wait_stream(stream)
    for _ in range(2):
        a.copy_(torch.randn(a.shape, device=a.device).to(a.dtype))
        b.copy_(torch.randn(b.shape, device=b.device).to(b.dtype))
        sa.mul_(1.3)
        sb.mul_(0.7)
        graph.replay()
        ref = _scaled_fp8_reference(a, b, sa, sb).to(out.dtype)
        torch.testing.assert_close(out, ref, rtol=0.016, atol=0.01)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _scaled_mm_backend_available(), reason="FP8 scaled MM backend")
def test_mm_w8a8_fp8_empty():
    for m, n, k in [(0, 32, 16), (32, 0, 16), (32, 16, 0)]:
        a = torch.empty((m, k), device=flag_gems.device, dtype=torch.float8_e4m3fn)
        b = torch.empty((k, n), device=flag_gems.device, dtype=a.dtype)
        sa, sb = _external_scales(a, b)
        result = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=torch.bfloat16)
        torch.testing.assert_close(
            result, torch.zeros((m, n), device=a.device, dtype=torch.bfloat16)
        )


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _scaled_mm_backend_available(), reason="FP8 scaled MM backend")
@pytest.mark.parametrize(
    "out_dtype", [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_mm_w8a8_fp8_output_dtype(out_dtype):
    a = torch.randn((64, 32), device=flag_gems.device).to(torch.float8_e4m3fn).T
    b = torch.randn((64, 32), device=flag_gems.device).to(a.dtype)
    sa, sb = _external_scales(a, b)
    ref = _scaled_fp8_reference(a, b, sa, sb)
    result = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=out_dtype)
    assert result.dtype == out_dtype
    torch.testing.assert_close(
        result.float(), ref.to(out_dtype).float(), rtol=0.001, atol=0.002
    )


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _scaled_mm_backend_available(), reason="FP8 scaled MM backend")
def test_mm_w8a8_fp8_independent_and_invalid_out():
    a = torch.randn((64, 32), device=flag_gems.device).to(torch.float8_e4m3fn).T
    b = torch.randn((64, 32), device=flag_gems.device).to(a.dtype)
    sa, sb = _external_scales(a, b)
    ref = _scaled_fp8_reference(a, b, sa, sb).to(torch.bfloat16)
    first = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=torch.bfloat16)
    second = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=torch.bfloat16)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, ref, rtol=0.016, atol=0.01)
    out = torch.empty((0,), device=a.device, dtype=torch.bfloat16)
    assert flag_gems.mm_w8a8_fp8_out(a, b, sa, sb, out=out) is out
    torch.testing.assert_close(out, ref, rtol=0.016, atol=0.01)
    with pytest.raises(ValueError, match="out_dtype must match out.dtype"):
        flag_gems.mm_w8a8_fp8_out(a, b, sa, sb, out_dtype=torch.float32, out=out)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _mthreads_w8a8_fp8_available(), reason="MThreads FP8 backend")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mm_w8a8_fp8_mthreads_broadcast(dtype):
    a = torch.randn((1, 64), device=flag_gems.device).to(dtype).expand(32, 64)
    b = torch.randn((64, 1), device=flag_gems.device).to(dtype).expand(64, 32)
    sa = torch.full((1,), 0.5, device=a.device).expand(32)
    sb = torch.full((1,), 1.5, device=b.device).expand(32)
    ref = _scaled_fp8_reference(a, b, sa, sb).to(torch.bfloat16)
    result = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=torch.bfloat16)
    torch.testing.assert_close(result, ref, rtol=0.016, atol=0.01)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _scaled_mm_backend_available(), reason="FP8 scaled MM backend")
@pytest.mark.parametrize("out_variant", [False, True])
def test_mm_w8a8_fp8_invalid_inputs_and_scales(out_variant):
    a = torch.randn((32, 64), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.randn((64, 16), device=flag_gems.device).to(a.dtype)
    sa, sb = _external_scales(a, b)
    if out_variant:
        op = flag_gems.mm_w8a8_fp8_out
        kwargs = {"out": torch.empty((32, 16), device=a.device, dtype=torch.bfloat16)}
    else:
        op = flag_gems.mm_w8a8_fp8
        kwargs = {}

    with pytest.raises(TypeError):
        op(a, b, **kwargs)
    with pytest.raises(TypeError):
        op(a, b, sa, **kwargs)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        with pytest.raises(TypeError, match="requires FP8 inputs"):
            op(a.to(dtype), b, sa, sb, **kwargs)
        with pytest.raises(TypeError, match="requires FP8 inputs"):
            op(a, b.to(dtype), sa, sb, **kwargs)

    for invalid in (None, 1.0, sa.to(torch.float16)):
        with pytest.raises(TypeError, match="scale_a must be a float32 tensor"):
            op(a, b, invalid, sb, **kwargs)
        with pytest.raises(TypeError, match="scale_b must be a float32 tensor"):
            op(a, b, sa, invalid, **kwargs)
    with pytest.raises(ValueError, match="scale_a must be on the same device"):
        op(a, b, sa.cpu(), sb, **kwargs)
    with pytest.raises(ValueError, match="scale_b must be on the same device"):
        op(a, b, sa, sb.cpu(), **kwargs)
    for shape in ((2,), (1, 32), (32, 2), (1, 1, 1)):
        invalid = torch.ones(shape, device=a.device)
        with pytest.raises(ValueError, match="scale_a must be a scalar"):
            op(a, b, invalid, sb, **kwargs)
    for shape in ((2,), (16, 1), (16, 2), (1, 1, 1)):
        invalid = torch.ones(shape, device=b.device)
        with pytest.raises(ValueError, match="scale_b must be a scalar"):
            op(a, b, sa, invalid, **kwargs)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _scaled_mm_backend_available(), reason="FP8 scaled MM backend")
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_mm_w8a8_fp8_scalar_shapes(shape):
    a = torch.randn((32, 128), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.randn((64, 128), device=flag_gems.device).to(a.dtype).T
    sa = torch.full(shape, 0.7, device=a.device)
    sb = torch.full(shape, 1.3, device=b.device)
    ref = _scaled_fp8_reference(a, b, sa, sb)
    result = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(result, ref, rtol=5e-4, atol=0.003)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _mthreads_w8a8_fp8_available(), reason="MThreads FP8 backend")
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (32, 64, 8192)])
@pytest.mark.parametrize(
    "out_dtype", [None, torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize("result_scale", ["none", "scalar", "row"])
@pytest.mark.parametrize("use_fast_accum", [False, True])
def test_mm_w8a8_fp8_mthreads_scaled_mm_interface(
    M, N, K, out_dtype, result_scale, use_fast_accum
):
    torch.manual_seed(42)
    a = torch.randn((M, K), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.randn((N, K), device=a.device).to(a.dtype).T
    sa, sb = _external_scales(a, b)
    bias_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    bias = torch.randn((N,), device=a.device, dtype=bias_dtype)
    sr = None
    if result_scale == "scalar":
        sr = torch.full((1,), 2.0, device=a.device)
    elif result_scale == "row":
        sr = torch.linspace(1.0, 2.0, M, device=a.device)
    kwargs = dict(
        input=a,
        mat2=b,
        scale_a=sa,
        scale_b=sb,
        bias=bias,
        scale_result=sr,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )
    expected = torch._scaled_mm(**kwargs)
    actual = flag_gems.mm_w8a8_fp8(**kwargs)
    assert actual.dtype == (a.dtype if out_dtype is None else out_dtype)
    ref = _scaled_fp8_reference(a, b, sa, sb) + bias.float()
    if sr is not None:
        ref /= sr.reshape(-1, 1)
    ref = ref.to(actual.dtype).float()
    rtol, atol = (0.125, 0.125) if out_dtype is None else (0.016, 0.01)
    if out_dtype == torch.float32:
        rtol, atol = 5e-4, 0.003
    torch.testing.assert_close(actual.float(), ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    out = torch.empty_like(actual)
    # Optional arguments retain the same positional order as aten::_scaled_mm.
    reused = flag_gems.mm_w8a8_fp8_out(
        a, b, sa, sb, bias, sr, out_dtype, use_fast_accum, out=out
    )
    assert reused is out
    torch.testing.assert_close(reused.float(), actual.float(), rtol=0, atol=0)
    assert flag_gems.mm_w8a8_fp8(**kwargs, out=out) is out
    torch.testing.assert_close(out.float(), actual.float(), rtol=0, atol=0)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _mthreads_w8a8_fp8_available(), reason="MThreads FP8 backend")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mm_w8a8_fp8_mthreads_default_dtype(dtype):
    a = torch.ones((16, 32), device=flag_gems.device).to(dtype)
    b = torch.ones((16, 32), device=a.device).to(dtype).T
    sa = torch.full((1,), 0.5, device=a.device)
    sb = torch.full((1,), 0.25, device=a.device)
    result = flag_gems.mm_w8a8_fp8(a, b, sa, sb)
    assert result.dtype == dtype
    torch.testing.assert_close(
        result.float(), torch.full((16, 16), 4.0, device=a.device)
    )


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _mthreads_w8a8_fp8_available(), reason="MThreads FP8 backend")
def test_mm_w8a8_fp8_mthreads_invalid_epilogue():
    a = torch.ones((16, 32), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.ones((32, 16), device=a.device).to(a.dtype)
    sa, sb = _external_scales(a, b)
    with pytest.raises(TypeError, match="use_fast_accum must be a bool"):
        flag_gems.mm_w8a8_fp8(a, b, sa, sb, use_fast_accum=1)
    with pytest.raises(TypeError, match="bias has an unsupported dtype"):
        flag_gems.mm_w8a8_fp8(a, b, sa, sb, bias=torch.ones(16, device=a.device))
    with pytest.raises(ValueError, match="bias must contain N elements"):
        flag_gems.mm_w8a8_fp8(
            a, b, sa, sb, bias=torch.ones(15, device=a.device).bfloat16()
        )
    with pytest.raises(ValueError, match="bias must contain N elements"):
        flag_gems.mm_w8a8_fp8(a, b, sa, sb, bias=torch.ones(16, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="scale_result must be a float32 tensor"):
        flag_gems.mm_w8a8_fp8(a, b, sa, sb, scale_result=2.0)
    with pytest.raises(ValueError, match="scale_result must be on the same device"):
        flag_gems.mm_w8a8_fp8(a, b, sa, sb, scale_result=torch.ones(1))
    with pytest.raises(ValueError, match="scale_result must be a scalar"):
        flag_gems.mm_w8a8_fp8(a, b, sa, sb, scale_result=torch.ones(2, device=a.device))


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _mthreads_w8a8_fp8_available(), reason="MThreads FP8 backend")
@pytest.mark.parametrize("K", [0, 128, 8192])
def test_mm_w8a8_fp8_mthreads_strided_epilogue(K):
    a = torch.ones((32, K), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.ones((64, K), device=a.device).to(a.dtype).T
    sa, sb = _external_scales(a, b)
    bias = torch.arange(128, device=a.device, dtype=torch.float32)[::2]
    sr = torch.linspace(1.0, 2.0, 64, device=a.device)[::2, None]
    out = torch.empty((32, 128), device=a.device)[:, ::2]
    ref = (_scaled_fp8_reference(a, b, sa, sb) + bias) / sr
    result = flag_gems.mm_w8a8_fp8(a, b, sa, sb, bias=bias, scale_result=sr, out=out)
    assert result is out
    torch.testing.assert_close(result, ref, rtol=5e-4, atol=0.003)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _cuda_hopper_w8a8_fp8_available(), reason="Hopper FP8 backend")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 1, 1),
        (7, 33, 15),
        (31, 65, 129),
        (32, 64, 7168),
        (128, 256, 8192),
        (192, 768, 8193),
        (256, 256, 16384),
        (256, 768, 1024),
        (512, 1024, 2048),
        (16, 9216, 2048),
        (129, 272, 6144),
        (256, 128, 16),
    ],
)
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("rowwise", [False, True])
@pytest.mark.parametrize("e5_operand", [None, "a", "b"])
def test_mm_w8a8_fp8_hopper_external_scales(M, N, K, column_major, rowwise, e5_operand):
    torch.manual_seed(42)
    ad = torch.float8_e5m2 if e5_operand == "a" else torch.float8_e4m3fn
    bd = torch.float8_e5m2 if e5_operand == "b" else torch.float8_e4m3fn
    a = torch.randn((M, K), device=flag_gems.device).to(ad)
    b = torch.randn((N, K) if column_major else (K, N), device=a.device).to(bd)
    if column_major:
        b = b.T
    sa, sb = _external_scales(a, b, rowwise)
    ref = _scaled_fp8_reference(a, b, sa, sb)
    out = torch.empty((M, N * 2), device=a.device, dtype=torch.bfloat16)[:, ::2]
    for _ in range(2):
        assert flag_gems.mm_w8a8_fp8_out(a, b, sa, sb, out=out) is out
        torch.testing.assert_close(out, ref.to(out.dtype), rtol=0.016, atol=0.01)
    result = flag_gems.mm_w8a8_fp8(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(result, ref, rtol=5e-4, atol=0.003)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _cuda_hopper_w8a8_fp8_available(), reason="Hopper FP8 backend")
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (256, 256, 8192)])
@pytest.mark.parametrize(
    "out_dtype", [None, torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize("with_scale_result", [False, True])
@pytest.mark.parametrize("use_fast_accum", [False, True])
def test_mm_w8a8_fp8_hopper_torch_interface(
    M, N, K, out_dtype, with_scale_result, use_fast_accum
):
    torch.manual_seed(42)
    a = torch.randn((M, K), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.randn((N, K), device=a.device).to(a.dtype).T
    sa, sb = _external_scales(a, b)
    bias = None
    if out_dtype != torch.float32:
        bias_dtype = torch.float16 if out_dtype == torch.float16 else torch.bfloat16
        bias = torch.randn((N,), device=a.device, dtype=bias_dtype)
    sr = torch.full((1,), 2.0, device=a.device) if with_scale_result else None
    kwargs = dict(
        input=a,
        mat2=b,
        scale_a=sa,
        scale_b=sb,
        bias=bias,
        scale_result=sr,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )
    # Our implementation retains FP32 promotion for both values of the hint.
    expected = torch._scaled_mm(**{**kwargs, "use_fast_accum": False})
    actual = flag_gems.mm_w8a8_fp8(**kwargs)
    assert actual.dtype == (a.dtype if out_dtype is None else out_dtype)
    ref = _scaled_fp8_reference(a, b, sa, sb)
    if bias is not None:
        ref += bias.float()
    ref = ref.to(actual.dtype).float()
    rtol, atol = (0.125, 0.125) if out_dtype is None else (0.016, 0.01)
    if out_dtype == torch.float32:
        rtol, atol = 5e-4, 0.003
    torch.testing.assert_close(actual.float(), ref, rtol=rtol, atol=atol)
    if out_dtype is None:
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=rtol, atol=atol
        )
    else:
        # cuBLAS FP8 accumulation differs near cancellation even with fast_accum=False.
        # Keep the strict elementwise check against the FP32 reference above.
        relative_error = torch.linalg.vector_norm(actual.float() - expected.float()) / (
            torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        )
        assert relative_error < (0.005 if out_dtype == torch.bfloat16 else 0.001)
    out = torch.empty_like(actual)
    assert flag_gems.mm_w8a8_fp8(**kwargs, out=out) is out
    torch.testing.assert_close(out.float(), actual.float(), rtol=rtol, atol=atol)
    assert (
        flag_gems.mm_w8a8_fp8_out(
            a, b, sa, sb, bias, sr, out_dtype, use_fast_accum, out=out
        )
        is out
    )
    torch.testing.assert_close(out.float(), actual.float(), rtol=rtol, atol=atol)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(not _cuda_hopper_w8a8_fp8_available(), reason="Hopper FP8 backend")
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (256, 256, 8192)])
def test_mm_w8a8_fp8_hopper_torch_rowwise(M, N, K):
    a = torch.randn((M, K), device=flag_gems.device).to(torch.float8_e4m3fn)
    b = torch.randn((N, K), device=a.device).to(a.dtype).T
    sa = torch.linspace(0.5, 1.0, M, device=a.device)[:, None]
    sb = torch.linspace(1.0, 1.5, N, device=a.device)[None, :]
    kwargs = dict(out_dtype=torch.bfloat16)
    expected = torch._scaled_mm(a, b, sa, sb, **kwargs)
    actual = flag_gems.mm_w8a8_fp8(a, b, sa, sb, **kwargs)
    ref = _scaled_fp8_reference(a, b, sa, sb).to(actual.dtype)
    torch.testing.assert_close(actual, ref, rtol=0.016, atol=0.01)
    relative_error = torch.linalg.vector_norm(actual.float() - expected.float()) / (
        torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    )
    assert relative_error < 0.005
