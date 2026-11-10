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

"""Accuracy tests for the ARM CPU INT8 linear stack and fused ops.

Covers the ARM64 CPU backend code:
  - pack_weights_sdot: SDOT tile layout contract and alignment checks
  - TLEInt8Linear: decode (TLE SDOT GEMV) and prefill (dynamic INT8 + _int_mm)
  - replace_linears_with_tle_int8 / quantize_and_replace_linears: module walk
  - apply_arm_overrides: curated key set, include/exclude filtering, idempotency
  - CUSTOMIZED_UNUSED_OPS vs the registered ops registry
  - arm_apply_rotary_pos_emb: NeoX / interleaved / inplace RoPE semantics
  - CPU torch_device_fn no-op guard patch
  - fused ops: arm_silu_and_mul, fused_add_rms_norm

Kernel-launching tests require the AArch64 triton-cpu backend (same condition
as tests/test_arm_w4a8_g128.py); the remaining contract tests only need the
TLE ops module to be importable.
"""

import platform

import pytest
import torch
import torch.nn.functional as F

import flag_gems

_ARM_CPU_BACKEND_ACTIVE = (
    platform.machine().lower() in {"arm64", "aarch64"}
    and flag_gems.vendor_name == "arm"
    and flag_gems.device == "cpu"
)

_SKIP_UNLESS_ARM_CPU = pytest.mark.skipif(
    not _ARM_CPU_BACKEND_ACTIVE,
    reason="requires an AArch64 Triton CPU backend",
)


def _import_int8():
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm.int8 import (
        TLEInt8Linear,
        pack_weights_sdot,
    )

    return TLEInt8Linear, pack_weights_sdot


def _fp32_linear_reference(x, w_int8, w_scale):
    effective = w_int8.float().t() * w_scale.float().unsqueeze(0)
    return x.float() @ effective


def test_pack_weights_sdot_layout():
    _, pack_weights_sdot = _import_int8()
    torch.manual_seed(0)
    K, N = 8, 12
    w = torch.randint(-128, 128, (K, N), dtype=torch.int8)
    packed = pack_weights_sdot(w)

    assert packed.shape == (K // 4, N // 4, 4, 4)
    assert packed.dtype == torch.int8
    assert packed.is_contiguous()

    # packed[a, b, c, d] must hold w[a * 4 + d, b * 4 + c]: each SDOT tile is
    # contiguous in memory (4 K-bytes x 4 N-lanes) for maximum L1 efficiency.
    a = torch.arange(K // 4).view(-1, 1, 1, 1)
    b = torch.arange(N // 4).view(1, -1, 1, 1)
    c = torch.arange(4).view(1, 1, -1, 1)
    d = torch.arange(4).view(1, 1, 1, -1)
    rows = (a * 4 + d).expand(K // 4, N // 4, 4, 4)
    cols = (b * 4 + c).expand(K // 4, N // 4, 4, 4)
    assert torch.equal(packed, w[rows, cols])


def test_pack_weights_sdot_rejects_misaligned():
    _, pack_weights_sdot = _import_int8()
    with pytest.raises(ValueError, match="K%4==0"):
        pack_weights_sdot(torch.zeros((5, 8), dtype=torch.int8))
    with pytest.raises(ValueError, match="N%4==0"):
        pack_weights_sdot(torch.zeros((8, 6), dtype=torch.int8))


def test_tle_int8_linear_constructor_contract():
    TLEInt8Linear, _ = _import_int8()
    torch.manual_seed(1)
    N, K = 8, 16
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8)
    s = torch.rand(N) * 0.05 + 1e-4

    lin = TLEInt8Linear(w, s)
    assert (lin.K, lin.N) == (K, N)
    assert lin._packed.shape == (K // 4, N // 4, 4, 4)
    assert torch.equal(lin._w_int8_kn, w.t().contiguous())
    assert torch.equal(lin._w_scale, s)

    scalar = TLEInt8Linear(w, torch.tensor(0.05))
    assert scalar._w_scale.numel() == 1

    with pytest.raises(TypeError, match="int8"):
        TLEInt8Linear(w.to(torch.float32), s)


@_SKIP_UNLESS_ARM_CPU
def test_tle_int8_linear_prefill_matches_reference():
    TLEInt8Linear, _ = _import_int8()
    torch.manual_seed(2)
    N, K = 64, 256
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8)
    s = torch.rand(N) * 0.01 + 1e-3
    lin = TLEInt8Linear(w, s)

    x = (0.1 * torch.randn(8, K)).to(torch.bfloat16)
    out = lin(x)

    assert out.shape == (8, N)
    assert out.dtype == torch.bfloat16
    # Tolerance covers the per-row dynamic INT8 activation quantization.
    ref = _fp32_linear_reference(x, w, s).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


@_SKIP_UNLESS_ARM_CPU
def test_tle_int8_linear_decode_matches_reference():
    TLEInt8Linear, _ = _import_int8()
    torch.manual_seed(3)
    N, K = 64, 512
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8)
    s = torch.rand(N) * 0.01 + 1e-3
    lin = TLEInt8Linear(w, s)

    x = (0.1 * torch.randn(1, K)).to(torch.bfloat16)
    out = lin(x)

    assert out.shape == (1, N)
    assert out.dtype == torch.bfloat16
    # Tolerance covers the in-kernel activation quantization.
    ref = _fp32_linear_reference(x, w, s).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


class _ToyModel(torch.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.fc = torch.nn.Linear(k, n, bias=False)
        self.keepfp = torch.nn.Linear(k, n, bias=False)
        self.head = torch.nn.Linear(k, n, bias=False)
        self.odd = torch.nn.Linear(k, 6, bias=False)
        self.noscale = torch.nn.Linear(k, n, bias=False)


@_SKIP_UNLESS_ARM_CPU
def test_replace_linears_with_tle_int8(monkeypatch):
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm import ops as arm_ops
    from flag_gems.runtime.backend._arm.int8 import (
        TLEInt8Linear,
        replace_linears_with_tle_int8,
    )

    # Keep the test hermetic: block the process-global aten::_int_mm override,
    # recording the request so the include contract is asserted too.
    override_calls = []
    monkeypatch.setattr(
        arm_ops,
        "apply_arm_overrides",
        lambda include=None, exclude=None: override_calls.append((include, exclude)),
    )

    torch.manual_seed(4)
    K, N = 16, 8
    model = _ToyModel(K, N)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8)
    s = torch.rand(N) * 0.05 + 1e-4
    state = {
        "fc.weight": w,
        "fc.weight_scale": s,
        "keepfp.weight": torch.randn(N, K),  # fp weight -> dtype skip
        "keepfp.weight_scale": s,
        "odd.weight": torch.randint(-128, 128, (6, K), dtype=torch.int8),
        "odd.weight_scale": torch.rand(6) * 0.05,  # N=6 -> alignment skip
        "noscale.weight": torch.randint(-128, 128, (N, K), dtype=torch.int8),
        # "noscale.weight_scale" missing -> missing skip
    }

    n_replaced = replace_linears_with_tle_int8(model, state, skip={"head"})

    assert n_replaced == 1
    assert isinstance(model.fc, TLEInt8Linear)
    for untouched in (model.keepfp, model.head, model.odd, model.noscale):
        assert isinstance(untouched, torch.nn.Linear)
    # The replacement pipeline must engage exactly the opt-in _int_mm override.
    assert override_calls == [(["_int_mm"], None)]

    # The replaced module must stay numerically faithful (prefill path).
    x = torch.randn(2, K)
    out = model.fc(x)
    ref = _fp32_linear_reference(x, w, s)
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)


@_SKIP_UNLESS_ARM_CPU
def test_quantize_and_replace_linears(monkeypatch):
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm import ops as arm_ops
    from flag_gems.runtime.backend._arm.int8 import (
        TLEInt8Linear,
        quantize_and_replace_linears,
    )

    monkeypatch.setattr(
        arm_ops,
        "apply_arm_overrides",
        lambda include=None, exclude=None: None,
    )

    torch.manual_seed(5)
    K, N = 16, 8
    model = _ToyModel(K, N)
    model.biased = torch.nn.Linear(K, N, bias=True)

    original_weight = model.fc.weight.data.clone()
    n_replaced = quantize_and_replace_linears(model, skip={"head"})

    # fc / keepfp / noscale are eligible; head explicit skip, odd alignment
    # skip, biased bias skip.
    assert n_replaced == 3
    for replaced in (model.fc, model.keepfp, model.noscale):
        assert isinstance(replaced, TLEInt8Linear)
    for untouched in (model.head, model.odd, model.biased):
        assert isinstance(untouched, torch.nn.Linear)

    # Per-channel symmetric INT8 quantization must dequantize back to the weight.
    # _w_int8_kn is [K, N] with a per-column [N] scale; dequantize to [N, K].
    dequant = model.fc._w_int8_kn.float().t() * model.fc._w_scale.unsqueeze(1)
    torch.testing.assert_close(dequant, original_weight.float(), rtol=1e-2, atol=5e-3)

    model2 = _ToyModel(K, N)
    model2.biased = torch.nn.Linear(K, N, bias=True)
    with pytest.raises(ValueError, match="bias"):
        quantize_and_replace_linears(model2, skip_with_bias=False)


def test_apply_arm_overrides_filters_and_is_idempotent(monkeypatch):
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm import ops as arm_ops

    calls = []

    def _make(name, raise_exc=False):
        def applier():
            if raise_exc:
                raise RuntimeError(name)
            calls.append(name)

        return applier

    registry = {
        "good_a": _make("good_a"),
        "good_b": _make("good_b"),
        "bad": _make("bad", raise_exc=True),
    }
    monkeypatch.setattr(arm_ops, "_ARM_OVERRIDE_REGISTRY", registry)
    monkeypatch.setattr(arm_ops, "_ARM_OVERRIDES_APPLIED", set())

    # Unknown names are ignored; exclude wins over include.
    arm_ops.apply_arm_overrides(
        include=["good_a", "good_b", "unknown"], exclude=["good_b"]
    )
    assert calls == ["good_a"]

    arm_ops.apply_arm_overrides()
    assert sorted(calls) == ["good_a", "good_b"]

    # Idempotent: a second full pass applies nothing new.
    arm_ops.apply_arm_overrides()
    assert sorted(calls) == ["good_a", "good_b"]

    # A failing applier must not be marked applied.
    arm_ops.apply_arm_overrides(include=["bad"])
    assert "bad" not in arm_ops._ARM_OVERRIDES_APPLIED


@_SKIP_UNLESS_ARM_CPU
def test_arm_silu_and_mul_matches_aten():
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm.ops.silu_and_mul import (
        arm_silu_and_mul,
        arm_silu_and_mul_out,
    )

    torch.manual_seed(6)
    # (512,) and (1, 512) hit the TLE SWIGLU decode kernel; (4, 512) hits ATen.
    for shape in [(512,), (1, 512), (4, 512)]:
        x1 = torch.randn(shape, dtype=torch.bfloat16)
        x2 = torch.randn(shape, dtype=torch.bfloat16)
        out = arm_silu_and_mul(x1, x2)
        ref = (F.silu(x1.float()) * x2.float()).to(torch.bfloat16)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-2)

    x1 = torch.randn(1, 256, dtype=torch.bfloat16)
    x2 = torch.randn(1, 256, dtype=torch.bfloat16)
    buf = torch.empty_like(x1)
    ret = arm_silu_and_mul_out(x1, x2, buf)
    assert ret is buf
    # Allow 1-ulp bf16 differences: the Triton exp lowering may round
    # differently from ATen's bf16 silu.
    torch.testing.assert_close(buf, F.silu(x1) * x2, rtol=1e-2, atol=0.02)


@_SKIP_UNLESS_ARM_CPU
def test_fused_add_rms_norm_matches_reference():
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
        fused_add_rms_norm,
    )

    torch.manual_seed(7)
    eps = 1e-5
    # (1, 896) is the decode shape; (3, 300) exercises the masked tail
    # (N is not a multiple of the 128-wide tile).
    for m, n in [(1, 896), (3, 300)]:
        x = (0.1 * torch.randn(m, n)).to(torch.bfloat16)
        residual = (0.1 * torch.randn(m, n)).to(torch.bfloat16)
        weight = torch.randn(n).to(torch.bfloat16)

        ref_residual = x.float() + residual.float()
        rrms = torch.rsqrt(ref_residual.pow(2).mean(-1, keepdim=True) + eps)
        ref_out = (ref_residual * rrms * weight.float()).to(torch.bfloat16)

        out, res = fused_add_rms_norm(x.clone(), residual.clone(), (n,), weight, eps)

        assert out.shape == (m, n)
        torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=1e-2)
        torch.testing.assert_close(res.float(), ref_residual, rtol=1e-3, atol=1e-3)


@_SKIP_UNLESS_ARM_CPU
def test_fused_add_rms_norm_updates_buffers_inplace():
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
        fused_add_rms_norm,
    )

    torch.manual_seed(8)
    m, n = 2, 256
    eps = 1e-5
    x = torch.randn(m, n, dtype=torch.bfloat16)
    residual = torch.randn(m, n, dtype=torch.bfloat16)
    weight = torch.randn(n, dtype=torch.bfloat16)

    # The kernel writes into the caller's buffers, so the reference must be
    # snapshotted before the call (x/residual are clobbered afterwards).
    x_snap = x.float()
    res_snap = residual.float()

    x_out, res_out = fused_add_rms_norm(x, residual, (n,), weight, eps)

    # Unlike flag_gems.add_rms_norm (out-of-place), the ARM kernel documents
    # an in-place contract: callers (vLLM forward_cpu) rely on the buffers
    # themselves being updated.
    assert x_out is x and res_out is residual
    ref_residual = x_snap + res_snap
    # The residual buffer is bf16, so allow a 1-ulp rounding difference at
    # full-scale randn magnitudes (bf16 ulp is ~0.004 relative).
    torch.testing.assert_close(res_out.float(), ref_residual, rtol=1e-2, atol=2e-2)


@_SKIP_UNLESS_ARM_CPU
def test_tle_int8_linear_decode_matches_prefill():
    """Both forward paths must agree: decode runs the TLE SDOT GEMV kernel,
    prefill runs per-row dynamic quant + torch._int_mm."""
    TLEInt8Linear, _ = _import_int8()
    torch.manual_seed(9)
    N, K = 64, 256
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8)
    s = torch.rand(N) * 0.01 + 1e-3
    lin = TLEInt8Linear(w, s)

    x = (0.1 * torch.randn(4, K)).to(torch.bfloat16)
    prefill = lin(x)
    decode = torch.cat([lin(x[i : i + 1].contiguous()) for i in range(4)])

    torch.testing.assert_close(decode, prefill, rtol=0.1, atol=0.1)


def test_arm_override_registry_is_curated():
    """The 8 override keys were hand-picked after benchmarking each op on ARM
    CPU (some candidates were dropped for regressing prefill). Adding or
    removing an entry must be a conscious decision, so pin the key set."""
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm import ops as arm_ops

    assert set(arm_ops._ARM_OVERRIDE_REGISTRY) == {
        "quantized_linear_dynamic",
        "_int_mm",
        "argmax",
        "fused_add_rms_norm",
        "apply_rotary_pos_emb",
        "silu_and_mul",
        "mm",
        "scaled_dot_product_attention",
    }


def test_customized_unused_ops_disjoint_from_registered_ops():
    """CUSTOMIZED_UNUSED_OPS is the auto-generated complement of the validated
    ops list. A registered name appearing there would silently unregister a
    kernel that was measured to beat ATen."""
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend import _arm
    from flag_gems.runtime.backend._arm import ops as arm_ops

    unused = set(_arm.CUSTOMIZED_UNUSED_OPS)
    registered = set(arm_ops.__all__)
    assert unused, "the exclusion list must not silently become empty"
    assert registered, "the ARM ops registry must not be empty"
    assert registered.isdisjoint(unused)


@pytest.mark.skipif(
    flag_gems.device != "cpu",
    reason="torch_device_fn is only patched on the CPU backend",
)
def test_cpu_torch_device_fn_guard_is_usable():
    """The CPU backend patches a no-op device()/._DeviceGuard onto
    torch_device_fn so flag_gems internals behave like other backends."""
    device_fn = flag_gems.runtime.torch_device_fn

    with device_fn.device():
        x = torch.zeros(4)
    assert x.shape == (4,)

    with device_fn._DeviceGuard():
        pass


# --- arm_apply_rotary_pos_emb: pure-PyTorch RoPE drop-in ---------------------


def _import_rope():
    pytest.importorskip(
        "triton.language.extra.cpu.tle_ops",
        reason="requires a triton-cpu build with TLE ops",
    )
    from flag_gems.runtime.backend._arm.ops.rope import arm_apply_rotary_pos_emb

    return arm_apply_rotary_pos_emb


def _rope_args(n_tokens=5, q_heads=4, kv_heads=2, rotary_dim=64, max_pos=16):
    torch.manual_seed(10)
    q = torch.randn(n_tokens, q_heads, rotary_dim)
    k = torch.randn(n_tokens, kv_heads, rotary_dim)
    half = rotary_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(half).float() / half))
    angles = torch.arange(max_pos).float()[:, None] * inv_freq[None, :]
    cos, sin = angles.cos(), angles.sin()
    pos_ids = torch.randperm(max_pos)[:n_tokens]
    return q, k, cos, sin, pos_ids


def _rope_neox_ref(x, cos, sin, pos_ids):
    c = cos[pos_ids][:, None, :]
    s = sin[pos_ids][:, None, :]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * torch.cat([c, c], dim=-1) + rotated * torch.cat([s, s], dim=-1)


def _rope_interleaved_ref(x, cos, sin, pos_ids):
    c = cos[pos_ids][:, None, :]
    s = sin[pos_ids][:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


@pytest.mark.parametrize("interleaved", [False, True])
def test_arm_apply_rotary_pos_emb_matches_reference(interleaved):
    arm_apply_rotary_pos_emb = _import_rope()
    q, k, cos, sin, pos_ids = _rope_args()
    torch_ref = _rope_interleaved_ref if interleaved else _rope_neox_ref

    q_out, k_out = arm_apply_rotary_pos_emb(
        q, k, cos, sin, position_ids=pos_ids, rotary_interleaved=interleaved
    )

    assert q_out.shape == q.shape and k_out.shape == k.shape
    torch.testing.assert_close(
        q_out, torch_ref(q, cos, sin, pos_ids), rtol=1e-5, atol=1e-5
    )
    # q and kv head counts differ; each input is rotated independently.
    torch.testing.assert_close(
        k_out, torch_ref(k, cos, sin, pos_ids), rtol=1e-5, atol=1e-5
    )


def test_arm_apply_rotary_pos_emb_optional_position_ids():
    arm_apply_rotary_pos_emb = _import_rope()
    q, k, cos, sin, pos_ids = _rope_args(n_tokens=3)

    # Without position_ids, cos/sin are consumed as pre-gathered [n, half].
    q_out, k_out = arm_apply_rotary_pos_emb(
        q, k, cos[pos_ids], sin[pos_ids], position_ids=None
    )

    torch.testing.assert_close(
        q_out, _rope_neox_ref(q, cos, sin, pos_ids), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        k_out, _rope_neox_ref(k, cos, sin, pos_ids), rtol=1e-5, atol=1e-5
    )


def test_arm_apply_rotary_pos_emb_inplace_writes_buffers():
    arm_apply_rotary_pos_emb = _import_rope()
    q, k, cos, sin, pos_ids = _rope_args()
    q_buf, k_buf = q.clone(), k.clone()

    q_out, k_out = arm_apply_rotary_pos_emb(
        q_buf, k_buf, cos, sin, position_ids=pos_ids, inplace=True
    )

    # vLLM's rope forward path requires the buffers themselves to be updated.
    assert torch.equal(q_out, q_buf) and torch.equal(k_out, k_buf)
    torch.testing.assert_close(
        q_buf, _rope_neox_ref(q, cos, sin, pos_ids), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        k_buf, _rope_neox_ref(k, cos, sin, pos_ids), rtol=1e-5, atol=1e-5
    )
