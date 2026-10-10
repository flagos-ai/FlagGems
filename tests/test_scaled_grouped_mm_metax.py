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

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pytest
import torch

import flag_gems

from .test_scaled_grouped_mm import _reference

pytestmark = [
    pytest.mark.scaled_grouped_mm,
    pytest.mark.skipif(
        flag_gems.runtime.device.vendor_name != "metax",
        reason="These cases exercise the MetaX mixed-precision contract",
    ),
]

MODES = ["m_varying", "n_varying", "k_varying", "batch"]
OUTPUT_DTYPES = [None, torch.float16, torch.bfloat16, torch.float32]
E4M3_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


@dataclass
class Inputs:
    a: torch.Tensor
    b: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    offs: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]


def _signed_tensor(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    result = torch.randint(-128, 128, shape, generator=generator, dtype=torch.int8)
    if result.numel() >= 2:
        result.reshape(-1)[:2] = torch.tensor([-128, 127], dtype=torch.int8)
    return result


def _binary_scales(shape: tuple[int, ...], exponent: int) -> torch.Tensor:
    count = 1
    for extent in shape:
        count *= extent
    # Powers of two preserve exact integer-dot results and avoid FP16 overflow.
    values = 2.0 ** ((torch.arange(count, dtype=torch.int32) % 3) - exponent)
    return values.to(torch.float32).reshape(shape)


def _make_cpu_inputs(
    mode: str,
    *,
    empty_groups: bool = False,
    zero_axis: Optional[str] = None,
    bias_kind: str = "grouped",
) -> Inputs:
    generator = torch.Generator(device="cpu").manual_seed(20261009)
    groups = 5 if empty_groups else 3
    m, n, k = 5, 7, 65
    if mode == "m_varying":
        sizes = [0, 2, 0, 3, 0] if empty_groups else [2, 3, 4]
        m = sum(sizes)
    elif mode == "n_varying":
        sizes = [0, 2, 0, 5, 0] if empty_groups else [2, 5, 4]
        n = sum(sizes)
    elif mode == "k_varying":
        sizes = [0, 1, 0, 64, 0] if empty_groups else [1, 32, 32]
    else:
        sizes = []

    if zero_axis == "M":
        m = 0
    elif zero_axis == "N":
        n = 0
    elif zero_axis == "K":
        k = 0
    if (mode, zero_axis) in (
        ("m_varying", "M"),
        ("n_varying", "N"),
        ("k_varying", "K"),
    ):
        sizes = [0] * groups

    a_shape = (m, k) if mode in ("m_varying", "k_varying") else (groups, m, k)
    b_shape = (k, n) if mode in ("n_varying", "k_varying") else (groups, k, n)
    if mode == "m_varying":
        sa_shape, sb_shape = (m,), (groups, n)
    elif mode == "n_varying":
        sa_shape, sb_shape = (groups, m), (n,)
    elif mode == "k_varying":
        sa_shape, sb_shape = (groups * m,), (groups * n,)
    else:
        sa_shape, sb_shape = (groups, m), (groups, n)

    offs = None
    if sizes:
        offs = torch.tensor(sizes, dtype=torch.int32).cumsum(0).to(torch.int32)
    bias = None
    if bias_kind != "none":
        bias_shape = (
            (n,) if bias_kind == "vector" or mode == "n_varying" else (groups, n)
        )
        count = n if len(bias_shape) == 1 else groups * n
        bias = ((torch.arange(count, dtype=torch.float32) % 9) - 4).reshape(
            bias_shape
        ) / 4
    return Inputs(
        _signed_tensor(a_shape, generator),
        _signed_tensor(b_shape, generator),
        _binary_scales(sa_shape, 7),
        _binary_scales(sb_shape, 5),
        offs,
        bias,
    )


def _strided_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
    storage_shape = (*tensor.shape[:-2], tensor.shape[-2] * 2, tensor.shape[-1] * 2)
    storage = torch.empty(storage_shape, dtype=tensor.dtype, device=flag_gems.device)
    result = storage[..., ::2, ::2]
    result.copy_(tensor.to(flag_gems.device))
    return result


def _to_device(inputs: Inputs, layout: str = "contiguous") -> Inputs:
    a = inputs.a.to(flag_gems.device)
    b = inputs.b.to(flag_gems.device)
    if layout == "strided_a_transposed_b":
        a = _strided_device_tensor(inputs.a)
        b = (
            inputs.b.transpose(-1, -2)
            .contiguous()
            .to(flag_gems.device)
            .transpose(-1, -2)
        )
        assert a.stride(-2) > 1 and a.stride(-1) > 1
        assert b.stride(-2) == 1
    elif layout == "strided_b":
        b = _strided_device_tensor(inputs.b)
        assert b.stride(-2) > 1 and b.stride(-1) > 1
    return Inputs(
        a,
        b,
        inputs.scale_a.to(flag_gems.device),
        inputs.scale_b.to(flag_gems.device),
        None if inputs.offs is None else inputs.offs.to(flag_gems.device),
        None if inputs.bias is None else inputs.bias.to(flag_gems.device),
    )


def _call(
    inputs: Inputs,
    out_dtype: Optional[torch.dtype] = torch.float32,
    *,
    scale_result: Optional[torch.Tensor] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    return flag_gems.scaled_grouped_mm(
        inputs.a,
        inputs.b,
        inputs.scale_a,
        inputs.scale_b,
        offs=inputs.offs,
        bias=inputs.bias,
        out_dtype=out_dtype,
        scale_result=scale_result,
        use_fast_accum=use_fast_accum,
    )


def _expected(inputs: Inputs, out_dtype: Optional[torch.dtype]) -> torch.Tensor:
    return _reference(
        inputs.a,
        inputs.b,
        inputs.scale_a,
        inputs.scale_b,
        inputs.offs,
        inputs.bias,
        out_dtype or torch.bfloat16,
    )


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_metax_public_entry_uses_backend_implementation() -> None:
    source = inspect.getsourcefile(flag_gems.scaled_grouped_mm)
    assert source is not None
    expected = (
        Path(flag_gems.__file__).resolve().parent
        / "runtime/backend/_metax/ops/scaled_grouped_mm.py"
    )
    assert Path(source).resolve() == expected, (
        "MetaX backend import/registration failed; testing the generic function "
        "would not validate the migrated implementation"
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn], ids=str)
def test_metax_public_entry_matches_reference(dtype: torch.dtype) -> None:
    cpu = _make_cpu_inputs("m_varying")
    cpu.a = cpu.a.to(dtype)
    cpu.b = cpu.b.to(dtype)
    inputs = _to_device(cpu)
    result = _call(inputs, torch.bfloat16)
    expected = _expected(cpu, torch.bfloat16)
    if dtype == torch.int8:
        _assert_exact(result, expected)
    else:
        # Tiny differences in the FP32 result can cross a BF16 halfway value.
        # The CPU comparison uses the repository's MetaX BF16 relative
        # error budget.
        torch.testing.assert_close(result.cpu(), expected, rtol=0.016, atol=1e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("out_dtype", OUTPUT_DTYPES, ids=str)
@pytest.mark.parametrize("layout", ["contiguous", "strided_a_transposed_b"])
def test_metax_int8_modes_outputs_and_layouts(
    mode: str, out_dtype: Optional[torch.dtype], layout: str
) -> None:
    # K=65 covers two complete tiles and a tail; full signed values distinguish
    # signed INT8 from unsigned storage or the MoE offset-128 packing contract.
    cpu = _make_cpu_inputs(mode)
    result = _call(_to_device(cpu, layout), out_dtype, use_fast_accum=True)
    _assert_exact(result, _expected(cpu, out_dtype))


@pytest.mark.parametrize("mode", MODES)
def test_metax_int8_strided_b(mode: str) -> None:
    cpu = _make_cpu_inputs(mode, bias_kind="none")
    _assert_exact(_call(_to_device(cpu, "strided_b")), _expected(cpu, torch.float32))


@pytest.mark.parametrize("mode", ["m_varying", "n_varying", "k_varying"])
@pytest.mark.parametrize("bias_kind", ["none", "grouped"])
def test_metax_int8_empty_groups(mode: str, bias_kind: str) -> None:
    cpu = _make_cpu_inputs(mode, empty_groups=True, bias_kind=bias_kind)
    _assert_exact(_call(_to_device(cpu)), _expected(cpu, torch.float32))


@pytest.mark.parametrize(
    "mode,zero_axis",
    [
        ("batch", "M"),
        ("batch", "N"),
        ("batch", "K"),
        ("m_varying", "M"),
        ("n_varying", "N"),
        ("k_varying", "K"),
    ],
)
def test_metax_int8_zero_extents(mode: str, zero_axis: str) -> None:
    cpu = _make_cpu_inputs(mode, zero_axis=zero_axis)
    _assert_exact(_call(_to_device(cpu)), _expected(cpu, torch.float32))


@pytest.mark.parametrize(
    "bias_dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str
)
@pytest.mark.parametrize("mode", ["m_varying", "batch", "k_varying"])
def test_metax_int8_vector_bias(mode: str, bias_dtype: torch.dtype) -> None:
    cpu = _make_cpu_inputs(mode, bias_kind="vector")
    assert cpu.bias is not None
    cpu.bias = cpu.bias.to(bias_dtype)
    _assert_exact(_call(_to_device(cpu)), _expected(cpu, torch.float32))


@pytest.mark.parametrize("dtype", [torch.int8, *E4M3_DTYPES], ids=str)
def test_metax_scale_before_fp16_cast(dtype: torch.dtype) -> None:
    # The unscaled result is 65536, but scaling in FP32 gives finite FP16 16384.
    a = torch.full((1, 3, 16), 64.0, dtype=torch.float32).to(dtype)
    b = torch.full((1, 16, 5), 64.0, dtype=torch.float32).to(dtype)
    cpu = Inputs(a, b, torch.full((1, 3), 0.5), torch.full((1, 5), 0.5), None, None)
    _assert_exact(
        _call(_to_device(cpu), torch.float16),
        torch.full((1, 3, 5), 16384.0, dtype=torch.float16),
    )


def test_metax_int8_rejects_accumulator_overflow() -> None:
    # Even (-128) * (-128) accumulated K times must fit signed INT32.
    k = 131072
    cpu = Inputs(
        torch.full((1, 1, k), -128, dtype=torch.int8),
        torch.full((1, k, 1), -128, dtype=torch.int8),
        torch.ones((1, 1)),
        torch.ones((1, 1)),
        None,
        None,
    )
    with pytest.raises(ValueError, match=r"K.*131071"):
        _call(_to_device(cpu))


def _force_predecode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Backend loading uses a dynamic module alias. Patching a second import
    # under flag_gems.runtime would not affect the public function's globals.
    module = sys.modules[flag_gems.scaled_grouped_mm.__module__]

    def enabled(m: int, n: int, k: int, num_groups: int, mode: int) -> bool:
        return True

    monkeypatch.setattr(module, "_use_predecode", enabled)


def _fp8_device_view(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    # Prepare bytes on CPU, then transfer storage and create views on device.
    # This avoids depending on native FP8 strided copy or quantization kernels.
    bits = tensor.view(torch.uint8)
    if layout == "column_major":
        storage = bits.transpose(-1, -2).contiguous().to(flag_gems.device)
        result = storage.transpose(-1, -2).view(tensor.dtype)
        assert result.stride(-2) == 1
    elif layout == "strided":
        shape = (*bits.shape[:-2], bits.shape[-2] * 2, bits.shape[-1] * 2)
        storage = torch.zeros(shape, dtype=torch.uint8)
        storage[..., ::2, ::2].copy_(bits)
        result = storage.to(flag_gems.device)[..., ::2, ::2].view(tensor.dtype)
        assert result.stride(-2) > 1 and result.stride(-1) > 1
    elif layout == "gapped_column_major":
        groups, rows, cols = bits.shape
        storage = torch.zeros((groups * 2, cols, rows), dtype=torch.uint8)
        storage[::2].copy_(bits.transpose(-1, -2))
        result = storage.to(flag_gems.device)[::2].transpose(-1, -2).view(tensor.dtype)
        assert result.stride(0) == 2 * rows * cols
        assert result.stride(-2) == 1
    else:
        assert layout == "contiguous"
        result = bits.to(flag_gems.device).view(tensor.dtype)
    return result


@pytest.mark.parametrize("dtype", E4M3_DTYPES, ids=str)
@pytest.mark.parametrize("mode", MODES)
def test_metax_predecode_modes_and_layouts(
    mode: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_predecode(monkeypatch)
    cpu = _make_cpu_inputs(
        mode,
        empty_groups=mode == "k_varying",
        bias_kind="vector" if mode == "batch" else "grouped",
    )
    # One-hot rows select K entries across tile boundaries and the final tail.
    # Binary fractions make this address/decoding check exact in FP32.
    a = torch.zeros(cpu.a.shape, dtype=torch.float32)
    rows = a.reshape(-1, a.shape[-1])
    for row in range(rows.shape[0]):
        rows[row, (0, 32, 64)[row % 3]] = 1
    b = (torch.arange(cpu.b.numel(), dtype=torch.float32) % 17 - 8).reshape(
        cpu.b.shape
    ) / 8
    cpu.a, cpu.b = a.to(dtype), b.to(dtype)
    layouts = {
        "m_varying": ("strided", "column_major"),
        "n_varying": ("column_major", "strided"),
        "k_varying": ("column_major", "contiguous"),
        "batch": ("contiguous", "gapped_column_major"),
    }
    a_layout, b_layout = layouts[mode]
    inputs = _to_device(cpu)
    inputs.a = _fp8_device_view(cpu.a, a_layout)
    inputs.b = _fp8_device_view(cpu.b, b_layout)
    _assert_exact(_call(inputs), _expected(cpu, torch.float32))


@pytest.mark.parametrize("dtype", E4M3_DTYPES, ids=str)
def test_metax_predecode_empty_k_with_bias(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_predecode(monkeypatch)
    cpu = _make_cpu_inputs("k_varying", zero_axis="K", bias_kind="grouped")
    cpu.a, cpu.b = cpu.a.to(dtype), cpu.b.to(dtype)
    _assert_exact(_call(_to_device(cpu)), _expected(cpu, torch.float32))


@pytest.mark.parametrize("dtype", E4M3_DTYPES, ids=str)
@pytest.mark.parametrize("predecode", [False, True])
def test_metax_e4m3_finite_encodings(
    dtype: torch.dtype, predecode: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if predecode:
        _force_predecode(monkeypatch)
    bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    if dtype == torch.float8_e4m3fnuz:
        bits[bits == 128] = 0
    else:
        bits[(bits & 127) == 127] = 0
    a = bits.view(dtype).reshape(1, 16, 16)
    b = torch.eye(16).to(dtype).reshape(1, 16, 16)
    cpu = Inputs(a, b, torch.ones((1, 16)), torch.ones((1, 16)), None, None)
    _assert_exact(_call(_to_device(cpu)), a.float())


@pytest.mark.parametrize("dtype", E4M3_DTYPES, ids=str)
@pytest.mark.parametrize("predecode", [False, True])
def test_metax_e4m3_nan_is_not_finite(
    dtype: torch.dtype, predecode: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if predecode:
        _force_predecode(monkeypatch)
    nan_bits = 128 if dtype == torch.float8_e4m3fnuz else 127
    a = torch.tensor([nan_bits, 0], dtype=torch.uint8).view(dtype).reshape(1, 2, 1)
    b = torch.ones((1, 1, 2)).to(dtype)
    cpu = Inputs(a, b, torch.ones((1, 2)), torch.ones((1, 2)), None, None)
    result = _call(_to_device(cpu)).cpu()
    assert torch.isnan(result[0, 0]).all()
    torch.testing.assert_close(result[0, 1], torch.zeros(2), rtol=0, atol=0)


def _noncontiguous(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.stack((tensor, tensor), dim=-1)[..., 0]
    assert not result.is_contiguous()
    return result


METADATA_ERRORS = [
    ("a_rank", RuntimeError, "2D or 3D"),
    ("b_rank", RuntimeError, "2D or 3D"),
    ("dtype_mismatch", RuntimeError, "same dtype"),
    ("k_mismatch", RuntimeError, "cannot be multiplied"),
    ("group_mismatch", RuntimeError, "batch sizes"),
    ("offs_missing", RuntimeError, "offs must be provided"),
    ("offs_dtype", RuntimeError, "int32"),
    ("offs_rank", RuntimeError, "1D"),
    ("offs_length", RuntimeError, "length"),
    ("offs_stride", ValueError, "contiguous"),
    ("scale_a_dtype", RuntimeError, "float32"),
    ("scale_a_shape", RuntimeError, "1D"),
    ("scale_b_shape", RuntimeError, "shape"),
    ("scale_a_stride", ValueError, "contiguous"),
    ("scale_b_stride", ValueError, "contiguous"),
    ("bias_shape", RuntimeError, "shape"),
    ("bias_dtype", TypeError, "bias"),
    ("bias_stride", ValueError, "contiguous"),
    ("n_grouped_bias", RuntimeError, "shape"),
    ("a_device", ValueError, "device"),
    ("b_device", ValueError, "device"),
    ("offs_device", ValueError, "device"),
    ("scale_a_device", ValueError, "device"),
    ("scale_b_device", ValueError, "device"),
    ("bias_device", ValueError, "device"),
    ("out_dtype", TypeError, "out_dtype"),
    ("scale_result", RuntimeError, "scale_result"),
]


@pytest.mark.parametrize("error,exception,match", METADATA_ERRORS)
def test_metax_mixed_metadata_errors(
    error: str, exception: type[Exception], match: str
) -> None:
    mode = "batch" if error == "group_mismatch" else "m_varying"
    if error == "n_grouped_bias":
        mode = "n_varying"
    inputs = _to_device(_make_cpu_inputs(mode))
    out_dtype = torch.float32
    scale_result = None
    if error == "a_rank":
        inputs.a = inputs.a.reshape(-1)
    elif error == "b_rank":
        inputs.b = inputs.b.reshape(-1)
    elif error == "dtype_mismatch":
        inputs.b = inputs.b.to(torch.float16)
    elif error == "k_mismatch":
        inputs.b = inputs.b[..., :-1, :]
    elif error == "group_mismatch":
        inputs.b = inputs.b[:2]
    elif error == "offs_missing":
        inputs.offs = None
    elif error == "offs_dtype":
        inputs.offs = inputs.offs.to(torch.int64)
    elif error == "offs_rank":
        inputs.offs = inputs.offs.unsqueeze(0)
    elif error == "offs_length":
        inputs.offs = inputs.offs[:2]
    elif error == "offs_stride":
        inputs.offs = _noncontiguous(inputs.offs)
    elif error == "scale_a_dtype":
        inputs.scale_a = inputs.scale_a.to(torch.float16)
    elif error == "scale_a_shape":
        inputs.scale_a = inputs.scale_a.unsqueeze(0)
    elif error == "scale_b_shape":
        inputs.scale_b = inputs.scale_b[:, :-1].contiguous()
    elif error == "scale_a_stride":
        inputs.scale_a = _noncontiguous(inputs.scale_a)
    elif error == "scale_b_stride":
        inputs.scale_b = _noncontiguous(inputs.scale_b)
    elif error == "bias_shape":
        inputs.bias = inputs.bias[:, :-1].contiguous()
    elif error == "bias_dtype":
        inputs.bias = inputs.bias.to(torch.int32)
    elif error == "bias_stride":
        inputs.bias = _noncontiguous(inputs.bias)
    elif error == "n_grouped_bias":
        inputs.bias = torch.ones((3, inputs.b.shape[-1]), device=flag_gems.device)
    elif error.endswith("_device"):
        name = error.removesuffix("_device")
        setattr(inputs, name, getattr(inputs, name).cpu())
    elif error == "out_dtype":
        out_dtype = torch.int8
    elif error == "scale_result":
        scale_result = torch.ones((), device=flag_gems.device)
    with pytest.raises(exception, match=match):
        _call(inputs, out_dtype=out_dtype, scale_result=scale_result)


def test_metax_int8_hot_path_avoids_host_reads_and_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu = _make_cpu_inputs("k_varying", empty_groups=True)
    inputs = _to_device(cpu, "strided_a_transposed_b")
    expected = _expected(cpu, torch.float32)
    # Warm the same signature before guarding the hot call, so compiler and
    # autotuner internals are outside the no-device-to-host-read assertion.
    _assert_exact(_call(inputs), expected)

    def guard(original: Callable[..., object]) -> Callable[..., object]:
        def wrapped(tensor: torch.Tensor, *args: object, **kwargs: object) -> object:
            if tensor.device.type != "cpu":
                raise AssertionError(
                    "The grouped GEMM hot path extracted a device tensor"
                )
            return original(tensor, *args, **kwargs)

        return wrapped

    original_to = torch.Tensor.to
    original_contiguous = torch.Tensor.contiguous

    def guarded_contiguous(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if tensor.device.type != "cpu" and not tensor.is_contiguous():
            raise AssertionError("The grouped GEMM hot path copied a strided tensor")
        return original_contiguous(tensor, *args, **kwargs)

    def guarded_to(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        target = kwargs.get("device", args[0] if args else None)
        if isinstance(target, torch.Tensor):
            target = target.device
        if isinstance(target, (str, torch.device)):
            if torch.device(target).type == "cpu" and tensor.device.type != "cpu":
                raise AssertionError(
                    "The grouped GEMM hot path copied a device tensor to CPU"
                )
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        for name in (
            "cpu",
            "item",
            "tolist",
            "numpy",
            "__bool__",
            "__int__",
            "__float__",
        ):
            patch.setattr(torch.Tensor, name, guard(getattr(torch.Tensor, name)))
        patch.setattr(torch.Tensor, "to", guarded_to)
        patch.setattr(torch.Tensor, "contiguous", guarded_contiguous)
        result = _call(inputs)
    _assert_exact(result, expected)
