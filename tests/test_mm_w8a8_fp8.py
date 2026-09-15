# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import triton

import flag_gems

from . import accuracy_utils as utils

pytestmark = [pytest.mark.mm_w8a8_fp8]


def _mod():
    return importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm_w8a8_fp8")


def _quant(x, dim):
    xf = x.float()
    scale = xf.abs().amax(dim=dim).clamp_min(1e-10) / 127.0
    view = scale[:, None] if dim == 1 else scale[None, :]
    return (xf / view).round().clamp(-128, 127).to(torch.int8), scale


def _ref(a, b):
    aq, sa = _quant(a, 1)
    bq, sb = _quant(b, 0)
    return ((aq.float() @ bq.float()) * sa[:, None] * sb[None, :]).to(a.dtype)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
def test_quantized_values_match_native(dtype, transpose):
    torch.manual_seed(321)
    a = torch.randn((256, 128) if transpose else (128, 256), device="npu", dtype=dtype)
    if transpose:
        a = a.t()
    q, s = _mod()._quantize_int8_rows(a)
    qr, sr = _quant(a, 1)
    torch.testing.assert_close(q, qr, rtol=0, atol=0)
    torch.testing.assert_close(s, sr, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(17, 48, 71), (65, 80, 129), (1, 16, 16)])
def test_tails_and_strided_out(dtype, shape):
    m, n, k = shape
    torch.manual_seed(234)
    a = torch.randn((k, m), device="npu", dtype=dtype).t()
    b = torch.randn((n, k), device="npu", dtype=dtype).t()
    mod = _mod()
    mod.clear_mm_w8a8_fp8_caches()
    out = torch.empty((m, n * 2), device="npu", dtype=dtype)[:, ::2]
    result = mod.mm_w8a8_fp8_out(a, b, out=out)
    assert result is out
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_weight_cache_inplace_mutation():
    mod = _mod()
    mod.clear_mm_w8a8_fp8_caches()
    torch.manual_seed(12)
    a = torch.randn((32, 128), device="npu", dtype=torch.bfloat16)
    b = torch.randn((128, 64), device="npu", dtype=torch.bfloat16)
    mod.mm_w8a8_fp8(a, b)
    b.neg_()
    out = mod.mm_w8a8_fp8(a, b)
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_inference_weight_cache_mutation():
    mod = _mod()
    with torch.inference_mode():
        a = torch.randn((16, 64), device="npu", dtype=torch.bfloat16)
        b = torch.randn((64, 32), device="npu", dtype=torch.bfloat16)
        mod.mm_w8a8_fp8(a, b)
        b.neg_()
        torch.testing.assert_close(
            mod.mm_w8a8_fp8(a, b), _ref(a, b), rtol=0.016, atol=0.0512
        )


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_graph_replay_observes_new_activation():
    mod = _mod()
    a = torch.randn((32, 128), device="npu", dtype=torch.bfloat16)
    b = torch.randn((128, 64), device="npu", dtype=torch.bfloat16)
    out = torch.empty((32, 64), device="npu", dtype=torch.bfloat16)
    for _ in range(3):
        mod.mm_w8a8_fp8_out(a, b, out=out)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        g.capture_begin()
        mod.mm_w8a8_fp8_out(a, b, out=out)
        g.capture_end()
    torch.npu.current_stream().wait_stream(stream)
    a.add_(0.25)
    g.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_two_streams_have_independent_accumulators():
    mod = _mod()
    a = [torch.randn((128, 256), device="npu", dtype=torch.bfloat16) for _ in range(2)]
    b = [torch.randn((256, 256), device="npu", dtype=torch.bfloat16) for _ in range(2)]
    streams = [torch.npu.Stream() for _ in range(2)]
    for i in range(2):
        mod.mm_w8a8_fp8(a[i], b[i])
    torch.npu.synchronize()
    outputs = []
    for i, stream in enumerate(streams):
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            outputs.append(mod.mm_w8a8_fp8(a[i], b[i]))
    torch.npu.synchronize()
    for i in range(2):
        torch.testing.assert_close(
            outputs[i], _ref(a[i], b[i]), rtol=0.016, atol=0.0512
        )


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_zero_rows_and_large_bfloat16_values():
    mod = _mod()
    a = torch.randn((16, 128), device="npu", dtype=torch.bfloat16) * 1000
    b = torch.randn((128, 64), device="npu", dtype=torch.bfloat16) * 1000
    a[0] = 0
    b[:, 0] = 0
    out = mod.mm_w8a8_fp8(a, b)
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)
    assert torch.isfinite(out).all().item()


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape,path",
    [
        ((1, 16, 16), "tiny_vector"),
        ((128, 256, 256), "int32_vector"),
        ((2048, 2048, 2048), "mixed"),
        ((2049, 2080, 2048), "mixed"),
    ],
)
def test_prepared_kernel_dispatch_and_tails(dtype, shape, path):
    m, n, k = shape
    torch.manual_seed(456)
    a = torch.randn((m, k), device="npu", dtype=dtype)
    b = torch.randn((k, n), device="npu", dtype=dtype)
    aq, sa = _quant(a, 1)
    bq, sb = _quant(b, 0)
    out = torch.empty((m, n), device="npu", dtype=dtype)
    call, meta = _mod()._prepare_mm_w8a8_kernel(aq, bq, sa, sb, out, m, n, k)
    assert meta["path"] == path
    call()
    torch.npu.synchronize()
    ref = ((aq.float() @ bq.float()) * sa[:, None] * sb[None, :]).to(dtype)
    torch.testing.assert_close(out, ref, rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("shape", [(1, 16, 16), (2048, 2048, 2048)])
def test_prepared_graph_reads_current_quantized_input(shape):
    m, n, k = shape
    torch.manual_seed(567)
    a = torch.randn((m, k), device="npu", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="npu", dtype=torch.bfloat16)
    aq, sa = _quant(a, 1)
    bq, sb = _quant(b, 0)
    out = torch.empty((m, n), device="npu", dtype=a.dtype)
    call, _ = _mod()._prepare_mm_w8a8_kernel(aq, bq, sa, sb, out, m, n, k)
    for _ in range(3):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph.capture_begin()
        call()
        graph.capture_end()
    torch.npu.current_stream().wait_stream(stream)
    aq.neg_()
    graph.replay()
    torch.npu.synchronize()
    ref = ((aq.float() @ bq.float()) * sa[:, None] * sb[None, :]).to(a.dtype)
    torch.testing.assert_close(out, ref, rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_mixed_workspace_is_safe_across_streams():
    torch.manual_seed(678)
    calls = []
    outputs = []
    refs = []
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    for _ in range(2):
        a = torch.randn((2048, 2048), device="npu", dtype=torch.bfloat16)
        b = torch.randn((2048, 2048), device="npu", dtype=torch.bfloat16)
        aq, sa = _quant(a, 1)
        bq, sb = _quant(b, 0)
        out = torch.empty_like(a)
        call, meta = _mod()._prepare_mm_w8a8_kernel(
            aq, bq, sa, sb, out, 2048, 2048, 2048
        )
        assert meta["path"] == "mixed"
        call()
        calls.append(call)
        outputs.append(out)
        refs.append(((aq.float() @ bq.float()) * sa[:, None] * sb[None, :]).to(a.dtype))
    torch.npu.synchronize()
    for stream, call in zip(streams, calls):
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            call()
    torch.npu.synchronize()
    for out, ref in zip(outputs, refs):
        torch.testing.assert_close(out, ref, rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_large_strided_out_uses_bounded_vector_tiles():
    torch.manual_seed(789)
    a = torch.randn((2048, 2048), device="npu", dtype=torch.bfloat16)
    b = torch.randn((2048, 2048), device="npu", dtype=torch.bfloat16)
    aq, sa = _quant(a, 1)
    bq, sb = _quant(b, 0)
    out = torch.empty((2048, 4096), device="npu", dtype=a.dtype)[:, ::2]
    call, meta = _mod()._prepare_mm_w8a8_kernel(aq, bq, sa, sb, out, 2048, 2048, 2048)
    assert meta["path"] == "int32_vector"
    assert meta["scale_tile"] == [4, 256]
    call()
    torch.npu.synchronize()
    ref = ((aq.float() @ bq.float()) * sa[:, None] * sb[None, :]).to(a.dtype)
    torch.testing.assert_close(out, ref, rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_static_output_with_input_tails(dtype):
    torch.manual_seed(901)
    a = torch.randn((48, 71), device="npu", dtype=dtype)
    b = torch.randn((71, 64), device="npu", dtype=dtype)
    aq, sa = _quant(a, 1)
    bq, sb = _quant(b, 0)
    out = torch.empty((48, 64), device="npu", dtype=dtype)
    fn, meta = _mod()._prepare_mm_w8a8_kernel(aq, bq, sa, sb, out, 48, 64, 71)
    assert meta["static_output"]
    fn()
    torch.npu.synchronize()
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (104, 256, 2048),  # Eight-row dense output tiles, not the static path.
        (136, 1024, 2048),
        (137, 80, 1024),  # New Cube tiling with masked M/N output tails.
        (8192, 64, 2048),  # Narrow-output mixed path.
        (8192, 256, 1024),
        (2048, 2048, 512),  # Short-K mixed path.
    ],
)
def test_profiled_dispatch_paths(dtype, shape):
    m, n, k = shape
    torch.manual_seed(5972)
    mod = _mod()
    mod.clear_mm_w8a8_fp8_caches()
    a = torch.randn((m, k), device="npu", dtype=dtype)
    b = torch.randn((k, n), device="npu", dtype=dtype)
    a[0] = 0
    b[:, 0] = 0
    aq, sa = _quant(a, 1)
    bq, sb = _quant(b, 0)
    out = torch.empty((m, n), device=a.device, dtype=dtype)
    call, _ = mod._prepare_mm_w8a8_kernel(aq, bq, sa, sb, out, m, n, k)
    call()
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_profiled_short_k_mixed_tail(dtype):
    test_profiled_dispatch_paths(dtype, (2051, 2080, 512))


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2048, 512),
        (48, 2048, 512),
        (400, 256, 2048),
        (333, 256, 2048),
        (1040, 1024, 1024),
        (24, 1024, 2048),
        (1, 12288, 2048),
        (48, 9216, 2048),
    ],
)
def test_profiled_tile_boundaries(dtype, shape):
    test_profiled_dispatch_paths(dtype, shape)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [(72, 12288, 2048), (136, 12288, 2048), (272, 9216, 2048), (368, 12288, 2048)],
)
def test_npot_cube_tiles(dtype, shape):
    test_profiled_dispatch_paths(dtype, shape)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [(104, 256, 2048), (160, 1024, 2048), (368, 2048, 512), (160, 12288, 2048)]
)
def test_ascendc_vector_epilogue(monkeypatch, dtype, shape):
    monkeypatch.setenv("FLAGGEMS_MM_W8A8_EPILOGUE", "ascendc")
    test_profiled_dispatch_paths(dtype, shape)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ascendc_vector_graph_updates(dtype):
    from flag_gems.runtime.backend._ascend.ops.ascendc.vector_epilogue import prepare

    c = torch.randint(-100000, 100000, (104, 256), device="npu", dtype=torch.int32)
    sa = torch.rand((104,), device="npu", dtype=torch.float32)
    sb = torch.rand((256,), device="npu", dtype=torch.float32)
    out = torch.empty((104, 256), device="npu", dtype=dtype)
    fn, _, _ = prepare(c, sa, sb, out, 104, 256, 8, 256)
    fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        g.capture_begin()
        fn()
        g.capture_end()
    torch.npu.current_stream().wait_stream(stream)
    c.neg_()
    sa.mul_(0.5)
    sb.mul_(0.25)
    g.replay()
    torch.npu.synchronize()
    ref = (c.float() * sa[:, None] * sb[None, :]).to(dtype)
    torch.testing.assert_close(out, ref, rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_custom_vector_output_abi():
    from flag_gems.runtime.backend._ascend.ops.ascendc.compile_fixpipe import (
        lower_custom_op_to_call,
    )

    ir = """module {
  func.func @test(%a: memref<8xf32>, %b: memref<8xf32>) {
CUSTOM_OPERATION_PLACEHOLDER
    return
  }
}"""
    ir = ir.replace(
        "CUSTOM_OPERATION_PLACEHOLDER",
        (
            '    hivm.hir.custom {symbol = "sample", hivm.tcore_type = #hivm.tcore_type<VECTO'
            'R>, extra_attr = "flaggems_pass_outputs=true"} "sample" ins(%a : memref<8xf32>) '
            "outs(%b : memref<8xf32>)"
        ),
    )
    lowered = lower_custom_op_to_call(ir)
    assert "func.func private @_mlir_ciface_sample(i64, i64)" in lowered
    assert "#hivm.func_core_type<AIV>" in lowered
    legacy = ir.replace(', extra_attr = "flaggems_pass_outputs=true"', "").replace(
        "tcore_type<VECTOR>", "tcore_type<CUBE>"
    )
    lowered = lower_custom_op_to_call(legacy)
    assert "func.func private @_mlir_ciface_sample(i64)" in lowered
    assert "#hivm.func_core_type<AIC>" in lowered


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [129, 160, 223, 224, 225, 256, 257, 368, 511, 512, 513])
def test_short_k_compact_tiles(dtype, m):
    test_profiled_dispatch_paths(dtype, (m, 2048, 512))


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [129, 144, 160, 176, 192, 224, 255, 256, 257])
def test_long_k_compact_tiles(dtype, m):
    test_profiled_dispatch_paths(dtype, (m, 2048, 4096))


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize(
    "shape",
    [
        (136, 64, 2048),
        (368, 64, 2048),
        (224, 256, 2048),
        (400, 256, 2048),
        (48, 2048, 512),
        (160, 2048, 512),
    ],
)
def test_aic_fpbuffer_paths(shape):
    test_profiled_dispatch_paths(torch.bfloat16, shape)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("scale", [1.0, 1000.0, 1e10])
def test_aic_fpbuffer_ranges(scale):
    mod = _mod()
    torch.manual_seed(193)
    a = torch.randn((136, 2048), device="npu", dtype=torch.bfloat16) * scale
    b = torch.randn((2048, 64), device="npu", dtype=a.dtype) * scale
    a[0] = 0
    b[:, 0] = 0
    out = mod.mm_w8a8_fp8(a, b)
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_aic_fpbuffer_graph_range_change():
    mod = _mod()
    torch.manual_seed(194)
    a = torch.randn((128, 2048), device="npu", dtype=torch.bfloat16)
    b = torch.randn((2048, 64), device="npu", dtype=a.dtype)
    out = torch.empty((128, 64), device=a.device, dtype=a.dtype)
    for _ in range(3):
        mod.mm_w8a8_fp8_out(a, b, out=out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph.capture_begin()
        mod.mm_w8a8_fp8_out(a, b, out=out)
        graph.capture_end()
    torch.npu.current_stream().wait_stream(stream)
    for factor in [1.0, 1e10]:
        a.mul_(factor)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_aic_fpbuffer_two_streams():
    torch.manual_seed(195)
    mod = _mod()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    inputs = [
        (
            torch.randn((136, 2048), device="npu", dtype=torch.bfloat16),
            torch.randn((2048, 64), device="npu", dtype=torch.bfloat16),
        )
        for _ in streams
    ]
    for a, b in inputs:
        mod.mm_w8a8_fp8(a, b)
    torch.npu.synchronize()
    outputs = []
    for stream, (a, b) in zip(streams, inputs):
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            outputs.append(mod.mm_w8a8_fp8(a, b))
    torch.npu.synchronize()
    for out, (a, b) in zip(outputs, inputs):
        torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_declared_cube_only_rejects_live_vector():
    from flag_gems.runtime.backend._ascend.ops.ascendc.compile_fixpipe import (
        strip_declared_cube_only_stub,
    )

    ir = """module {
  func.func @sample_mix_aic() attributes {hivm.part_of_mix} {
    return
  }
  func.func @sample_mix_aiv() attributes {hivm.part_of_mix} {
    return
  }
}"""
    result = strip_declared_cube_only_stub(ir)
    assert "@sample(" in result
    assert "@sample_mix_aiv" not in result
    assert "hivm.part_of_mix" not in result
    for op in [
        "func.call @write_output() : () -> ()",
        "hivm.hir.store ins(%a : memref<16xf32>) outs(%b : memref<16xf32>)",
    ]:
        bad = ir.replace(
            "  func.func @sample_mix_aiv() attributes {hivm.part_of_mix} {\n    return",
            "  func.func @sample_mix_aiv() attributes {hivm.part_of_mix} {\n    "
            + op
            + "\n    return",
        )
        with pytest.raises(ValueError):
            strip_declared_cube_only_stub(bad)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize(
    "shape", [(48, 1024, 2048), (272, 2048, 512), (352, 2048, 512), (480, 2048, 512)]
)
def test_aic_extended_dispatch(shape):
    test_profiled_dispatch_paths(torch.bfloat16, shape)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [257, 304, 384, 448])
def test_n1024_balanced_tiles(dtype, m):
    test_profiled_dispatch_paths(dtype, (m, 1024, 2048))


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (7, 12288, 2048),
        (8, 12288, 2048),
        (24, 12288, 2048),
        (31, 12288, 2048),
        (32, 12288, 2048),
        (33, 12288, 2048),
        (32, 9216, 2048),
        (33, 9216, 2048),
        (40, 9216, 2048),
        (48, 9216, 2048),
        (56, 9216, 2048),
        (64, 9216, 2048),
        (65, 9216, 2048),
        (128, 9216, 2048),
        (496, 12288, 2048),
    ],
)
def test_wide_dense_output_boundaries(dtype, shape):
    test_profiled_dispatch_paths(dtype, shape)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nz_graph_input_updates(dtype):
    mod = _mod()
    torch.manual_seed(911)
    a = torch.randn((40, 2048), device="npu", dtype=dtype)
    b = torch.randn((2048, 9216), device="npu", dtype=dtype)
    out = torch.empty((40, 9216), device=a.device, dtype=dtype)
    for _ in range(3):
        mod.mm_w8a8_fp8_out(a, b, out=out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph.capture_begin()
        mod.mm_w8a8_fp8_out(a, b, out=out)
        graph.capture_end()
    torch.npu.current_stream().wait_stream(stream)
    for factor in [-0.5, 3.0]:
        a.mul_(factor)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_nz_two_streams():
    mod = _mod()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    inputs = [
        (
            torch.randn((24, 2048), device="npu", dtype=torch.bfloat16),
            torch.randn((2048, 12288), device="npu", dtype=torch.bfloat16),
        )
        for _ in streams
    ]
    for a, b in inputs:
        mod.mm_w8a8_fp8(a, b)
    torch.npu.synchronize()
    outputs = []
    for st, (a, b) in zip(streams, inputs):
        st.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(st):
            outputs.append(mod.mm_w8a8_fp8(a, b))
    torch.npu.synchronize()
    for out, (a, b) in zip(outputs, inputs):
        torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_nz_large_range():
    mod = _mod()
    a = torch.randn((48, 2048), device="npu", dtype=torch.bfloat16) * 1e10
    b = torch.randn((2048, 9216), device="npu", dtype=torch.bfloat16) * 1e10
    out = mod.mm_w8a8_fp8(a, b)
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_nz_dispatch_preserves_fp32_output():
    mod = _mod()
    a = torch.randn((40, 2048), device="npu", dtype=torch.bfloat16)
    b = torch.randn((2048, 9216), device="npu", dtype=torch.bfloat16)
    aq, sa = _quant(a, 1)
    bq, sb = _quant(b, 0)
    out = torch.empty((40, 9216), device=a.device, dtype=torch.float32)
    fn, meta = mod._prepare_mm_w8a8_kernel(aq, bq, sa, sb, out, 40, 9216, 2048)
    assert meta["path"] == "int32_vector"
    fn()
    ref = (aq.float() @ bq.float()) * sa[:, None] * sb[None, :]
    torch.testing.assert_close(out, ref, rtol=0.016, atol=0.0512)
    public_out = mod.mm_w8a8_fp8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(public_out, ref, rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_nz_dispatch_preserves_strided_output():
    mod = _mod()
    a = torch.randn((40, 2048), device="npu", dtype=torch.bfloat16)
    b = torch.randn((2048, 9216), device="npu", dtype=torch.bfloat16)
    out = torch.empty((40, 9216 * 2), device=a.device, dtype=a.dtype)[:, ::2]
    result = mod.mm_w8a8_fp8_out(a, b, out=out)
    assert result is out
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [65, 72, 80, 128, 129, 256, 512, 513])
def test_nz_n1024_boundaries(dtype, m):
    test_profiled_dispatch_paths(dtype, (m, 1024, 2048))


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1024, 2048, 4096, 8184, 8192])
def test_nz_n256_boundaries(dtype, m):
    test_profiled_dispatch_paths(dtype, (m, 256, 2048))


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
def test_aic_nz_large_range():
    mod = _mod()
    a = torch.randn((256, 512), device="npu", dtype=torch.bfloat16) * 1e10
    b = torch.randn((512, 2048), device="npu", dtype=torch.bfloat16) * 1e10
    out = mod.mm_w8a8_fp8(a, b)
    torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("m", [256, 2048])
def test_aic_nz_graph_range_change(m):
    mod = _mod()
    a = torch.randn((m, 512), device="npu", dtype=torch.bfloat16)
    b = torch.randn((512, 2048), device="npu", dtype=torch.bfloat16)
    out = torch.empty((m, 2048), device=a.device, dtype=a.dtype)
    for _ in range(3):
        mod.mm_w8a8_fp8_out(a, b, out=out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph.capture_begin()
        mod.mm_w8a8_fp8_out(a, b, out=out)
        graph.capture_end()
    torch.npu.current_stream().wait_stream(stream)
    for factor in [1.0, 1e10]:
        a.mul_(factor)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("m", [336, 2048])
def test_aic_nz_two_streams(m):
    mod = _mod()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    inputs = [
        (
            torch.randn((m, 512), device="npu", dtype=torch.bfloat16),
            torch.randn((512, 2048), device="npu", dtype=torch.bfloat16),
        )
        for _ in streams
    ]
    for a, b in inputs:
        mod.mm_w8a8_fp8(a, b)
    torch.npu.synchronize()
    outputs = []
    for st, (a, b) in zip(streams, inputs):
        st.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(st):
            outputs.append(mod.mm_w8a8_fp8(a, b))
    torch.npu.synchronize()
    for out, (a, b) in zip(outputs, inputs):
        torch.testing.assert_close(out, _ref(a, b), rtol=0.016, atol=0.0512)


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend W8A8 regression")
@pytest.mark.parametrize("m", [1024, 1040, 2048, 16368, 16384, 16400])
def test_aic_batch_prefetch_boundaries(m):
    test_profiled_dispatch_paths(torch.bfloat16, (m, 2048, 512))


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


def _mm_w8a8_fp8_available():
    # Hopper: FP8 e4m3 + TMA. Ascend: INT8 + per-row/col scale (UB cannot load FP8).
    return flag_gems.vendor_name == "ascend" or _cuda_hopper_w8a8_fp8_available()


def _mm_w8a8_fp8_reference(a, b):
    if flag_gems.vendor_name == "ascend":
        a_fp32 = a.float()
        a_scale = a_fp32.abs().amax(dim=1).clamp_min(1e-10) / 127
        a_q = (a_fp32 / a_scale[:, None]).round().clamp(-128, 127).to(torch.int8)
        b_fp32 = b.float()
        b_scale = b_fp32.abs().amax(dim=0).clamp_min(1e-10) / 127
        b_q = (b_fp32 / b_scale[None, :]).round().clamp(-128, 127).to(torch.int8)
        return torch.mm(a_q.float(), b_q.float()) * a_scale[:, None] * b_scale[None, :]
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    a_fp32 = a.float()
    a_scale = a_fp32.abs().amax(dim=1).clamp_min(1e-10) / fp8_info.max
    a_fp8 = (a_fp32 / a_scale[:, None]).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
    b_fp32 = b.float()
    b_scale = b_fp32.abs().amax(dim=0).clamp_min(1e-10) / fp8_info.max
    b_fp8 = (b_fp32 / b_scale[None, :]).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
    return torch.mm(a_fp8.float(), b_fp8.float()) * a_scale[:, None] * b_scale[None, :]


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
    not _mm_w8a8_fp8_available(),
    reason="mm_w8a8_fp8 requires Ascend or CUDA Hopper FP8 and TMA support",
)
def test_mm_w8a8_fp8(M, N, K):
    dtype = torch.bfloat16
    torch.manual_seed(0)

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_out = utils.to_reference(_mm_w8a8_fp8_reference(mat1, mat2), True)

    res_out = flag_gems.mm_w8a8_fp8(mat1, mat2, out_dtype=dtype)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    res_out_reused = flag_gems.mm_w8a8_fp8_out(mat1, mat2, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    utils.gems_assert_close(res_out_reused, ref_out, dtype, reduce_dim=K)
