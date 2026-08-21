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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

METAX_SILU_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
METAX_SILU_GRID_BALANCE_SHAPES = (
    (1024, 131073),
    (64, 64, 40961),
    (1024, 262145),
    (64, 64, 81921),
    (1024, 393217),
    (64, 64, 131073),
)
METAX_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="MetaX override"
)


SILU_ASCEND_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SILU_EDGE_CASES = ["tail", "non_contiguous", "empty", "special_values"]
ASCEND_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-only SiLU coverage"
)


def _make_silu_edge_input(case, dtype):
    if case == "tail":
        return torch.linspace(
            -8.0,
            8.0,
            4097,
            dtype=torch.float32,
            device=flag_gems.device,
        ).to(dtype)
    if case == "non_contiguous":
        return torch.randn((257, 259), dtype=dtype, device=flag_gems.device).transpose(
            0, 1
        )
    if case == "empty":
        return torch.empty((0, 17), dtype=dtype, device=flag_gems.device)
    if case == "special_values":
        return torch.tensor(
            [
                float("-inf"),
                -20.0,
                -1.0,
                -0.0,
                0.0,
                1.0,
                20.0,
                float("inf"),
                float("nan"),
            ],
            dtype=dtype,
            device=flag_gems.device,
        )
    raise AssertionError(f"unsupported SiLU edge case: {case}")


@pytest.mark.silu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu
@pytest.mark.silu_
@ASCEND_ONLY
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("case", SILU_EDGE_CASES)
@pytest.mark.parametrize("dtype", SILU_ASCEND_DTYPES)
def test_silu_edge_cases(case, dtype, inplace):
    res_inp = _make_silu_edge_input(case, dtype)
    ref_inp = utils.to_reference(res_inp.clone(), True)
    original_stride = res_inp.stride()
    original_storage_ptr = res_inp.untyped_storage().data_ptr()
    native_out = None
    if case == "non_contiguous" and not inplace:
        native_out = torch.nn.functional.silu(res_inp)

    ref_out = torch.nn.functional.silu(ref_inp, inplace=inplace)
    selected_ops = ["silu_"] if inplace else ["silu"]
    with flag_gems.use_gems(include=selected_ops):
        res_out = torch.nn.functional.silu(res_inp, inplace=inplace)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    if native_out is not None:
        assert res_out.layout == native_out.layout
        assert res_out.stride() == native_out.stride()
        assert res_out.is_contiguous() == native_out.is_contiguous()
    if inplace:
        assert res_out.untyped_storage().data_ptr() == original_storage_ptr
        assert res_out.stride() == original_stride


@pytest.mark.silu
@pytest.mark.silu_backward
@ASCEND_ONLY
@pytest.mark.parametrize("dtype", SILU_ASCEND_DTYPES)
def test_silu_forward_autograd(dtype):
    res_inp = torch.linspace(
        -4.0,
        4.0,
        4097,
        dtype=torch.float32,
        device=flag_gems.device,
    ).to(dtype)
    res_inp.requires_grad_(True)
    res_grad_out = torch.linspace(
        0.25,
        1.25,
        4097,
        dtype=torch.float32,
        device=flag_gems.device,
    ).to(dtype)
    ref_inp = utils.to_reference(res_inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(res_grad_out, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)
    with flag_gems.use_gems(include=["silu", "silu_backward"]):
        res_out = torch.nn.functional.silu(res_inp)
        (res_grad,) = torch.autograd.grad(res_out, res_inp, res_grad_out)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.silu_
@ASCEND_ONLY
@pytest.mark.parametrize("dtype", SILU_ASCEND_DTYPES)
def test_silu_inplace_leaf_autograd_error(dtype):
    res_inp = torch.randn(
        (4097,), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_inp = utils.to_reference(res_inp.detach(), True).requires_grad_(True)

    with pytest.raises(RuntimeError):
        torch.nn.functional.silu(ref_inp, inplace=True)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems(include=["silu_"]):
            torch.nn.functional.silu(res_inp, inplace=True)


@pytest.mark.silu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.nn.functional.silu(ref_inp, inplace=True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp, inplace=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.silu_backward(ref_grad, ref_inp)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.silu_backward(res_grad, res_inp)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.silu
@METAX_ONLY
@pytest.mark.parametrize("dtype", METAX_SILU_DTYPES)
def test_metax_silu_forward_contract(dtype):
    inp = torch.randn((2, 19, 7), dtype=dtype, device=flag_gems.device)
    original = inp.clone()
    reference = torch.nn.functional.silu(utils.to_reference(inp, True))

    with flag_gems.use_gems(include=["silu"]):
        actual = torch.nn.functional.silu(inp)

    utils.gems_assert_close(actual, reference, dtype)
    assert actual.shape == inp.shape
    assert actual.dtype == inp.dtype
    torch.testing.assert_close(inp, original)


@pytest.mark.silu
@METAX_ONLY
@pytest.mark.parametrize("shape", METAX_SILU_GRID_BALANCE_SHAPES)
@pytest.mark.parametrize("dtype", METAX_SILU_DTYPES)
def test_metax_silu_grid_balance_shapes(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    reference = torch.nn.functional.silu(utils.to_reference(inp, True))

    with flag_gems.use_gems(include=["silu"]):
        actual = torch.nn.functional.silu(inp)

    utils.gems_assert_close(actual, reference, dtype)


@pytest.mark.silu
@METAX_ONLY
@pytest.mark.parametrize("dtype", METAX_SILU_DTYPES)
def test_metax_silu_empty(dtype):
    inp = torch.empty((0, 19, 7), dtype=dtype, device=flag_gems.device)
    reference = torch.nn.functional.silu(utils.to_reference(inp, True))

    with flag_gems.use_gems(include=["silu"]):
        actual = torch.nn.functional.silu(inp)

    utils.gems_assert_close(actual, reference, dtype)
    assert actual.shape == inp.shape
    assert actual.dtype == inp.dtype


@pytest.mark.silu
@METAX_ONLY
@pytest.mark.parametrize("dtype", METAX_SILU_DTYPES)
def test_metax_silu_special_values(dtype):
    inp = torch.tensor(
        [
            float("-inf"),
            -17.0,
            -0.0,
            0.0,
            17.0,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    reference = torch.nn.functional.silu(utils.to_reference(inp, True))

    with flag_gems.use_gems(include=["silu"]):
        actual = torch.nn.functional.silu(inp)

    utils.gems_assert_close(actual, reference, dtype, equal_nan=True)


@pytest.mark.silu
@METAX_ONLY
def test_metax_silu_noncontiguous_uses_common_fallback(monkeypatch):
    module = importlib.import_module("_metax.ops.silu")
    real_generic_silu = module._generic_silu
    calls = []

    def tracked_generic_silu(inp):
        calls.append((inp.shape, inp.stride(), inp.dtype))
        return real_generic_silu(inp)

    monkeypatch.setattr(module, "_generic_silu", tracked_generic_silu)
    inp = torch.randn((19, 7), dtype=torch.float32, device=flag_gems.device).t()
    reference = torch.nn.functional.silu(utils.to_reference(inp, True))

    with flag_gems.use_gems(include=["silu"]):
        actual = torch.nn.functional.silu(inp)

    assert calls == [(inp.shape, inp.stride(), inp.dtype)]
    utils.gems_assert_close(actual, reference, inp.dtype)


@pytest.mark.silu
@METAX_ONLY
def test_metax_silu_public_path_record(tmp_path):
    record_path = tmp_path / "metax_silu.log"
    contiguous = torch.randn((2, 19, 7), dtype=torch.float32, device=flag_gems.device)
    noncontiguous = torch.randn(
        (19, 7), dtype=torch.float32, device=flag_gems.device
    ).t()

    with flag_gems.use_gems(include=["silu"], record=True, path=str(record_path)):
        contiguous_out = torch.nn.functional.silu(contiguous)
        noncontiguous_out = torch.nn.functional.silu(noncontiguous)

    assert contiguous_out.shape == contiguous.shape
    assert noncontiguous_out.shape == noncontiguous.shape
    assert record_path.exists()
    log_content = record_path.read_text()
    assert log_content.count("GEMS_METAX SILU FORWARD") == 1
    assert log_content.count("GEMS SILU FORWARD") == 1
    assert log_content.index("GEMS_METAX SILU FORWARD") < log_content.index(
        "GEMS SILU FORWARD"
    )


@pytest.mark.silu_
@METAX_ONLY
@pytest.mark.parametrize("dtype", METAX_SILU_DTYPES)
def test_metax_silu_inplace_keeps_alias(dtype):
    inp = torch.randn((2, 19, 7), dtype=dtype, device=flag_gems.device)
    reference = torch.nn.functional.silu(utils.to_reference(inp, True))
    original_data_ptr = inp.data_ptr()

    with flag_gems.use_gems(include=["silu_"]):
        actual = torch.nn.functional.silu(inp, inplace=True)

    assert actual.data_ptr() == original_data_ptr
    assert inp.data_ptr() == original_data_ptr
    utils.gems_assert_close(actual, reference, dtype)


@pytest.mark.silu_backward
@METAX_ONLY
@pytest.mark.parametrize("dtype", METAX_SILU_DTYPES)
def test_metax_silu_direct_backward_stays_common(dtype):
    inp = torch.randn((2, 19, 7), dtype=dtype, device=flag_gems.device)
    grad = torch.randn_like(inp)
    reference = torch.ops.aten.silu_backward(
        utils.to_reference(grad, True), utils.to_reference(inp, True)
    )

    with flag_gems.use_gems(include=["silu_backward"]):
        actual = torch.ops.aten.silu_backward(grad, inp)

    utils.gems_assert_close(actual, reference, dtype)


@pytest.mark.silu
@METAX_ONLY
def test_metax_silu_registrar_scope():
    common_module = importlib.import_module("flag_gems.ops.silu")

    assert flag_gems.silu.__module__ == "_metax.ops.silu"
    metax_module = importlib.import_module(flag_gems.silu.__module__)
    assert flag_gems.silu is metax_module.silu
    assert flag_gems.silu_ is common_module.silu_
    assert flag_gems.silu_backward is common_module.silu_backward


@pytest.mark.silu
@METAX_ONLY
@pytest.mark.parametrize(
    ("dtype", "numel"),
    (
        (torch.float16, (1 << 24) - 1),
        (torch.float16, 1 << 24),
        (torch.float16, (1 << 24) + 1),
        (torch.bfloat16, (1 << 24) - 1),
        (torch.bfloat16, 1 << 24),
        (torch.bfloat16, (1 << 24) + 1),
        (torch.float32, (1 << 24) - 1),
        (torch.float32, 1 << 24),
        (torch.float32, (1 << 24) + 1),
    ),
)
def test_metax_silu_supported_contiguous_uses_single_route(dtype, numel, monkeypatch):
    module = importlib.import_module("_metax.ops.silu")
    routes = []

    def tracked_forward(inp):
        routes.append("balanced")
        return inp

    def forbidden_large_route(_):
        pytest.fail("supported contiguous SiLU must not use a dtype/size tier")

    monkeypatch.setattr(module, "silu_forward", tracked_forward)
    if hasattr(module, "silu_forward_large"):
        monkeypatch.setattr(module, "silu_forward_large", forbidden_large_route)
    inp = torch.empty(numel, dtype=dtype, device=flag_gems.device)
    assert inp.is_contiguous()

    with flag_gems.use_gems(include=["silu"]):
        actual = torch.nn.functional.silu(inp)

    assert actual is inp
    assert routes == ["balanced"]


@pytest.mark.silu
@METAX_ONLY
def test_metax_silu_large_noncontiguous_uses_generic_fallback(monkeypatch):
    module = importlib.import_module("_metax.ops.silu")
    routes = []

    def tracked_supported(inp):
        routes.append("balanced")
        return inp

    def tracked_generic(inp):
        routes.append("generic")
        return inp

    monkeypatch.setattr(module, "silu_forward", tracked_supported)
    monkeypatch.setattr(module, "_generic_silu", tracked_generic)
    inp = torch.empty(
        (2, (1 << 24) // 2),
        dtype=torch.float16,
        device=flag_gems.device,
    ).t()

    with flag_gems.use_gems(include=["silu"]):
        actual = torch.nn.functional.silu(inp)

    assert actual is inp
    assert not inp.is_contiguous()
    assert inp.numel() == 1 << 24
    assert routes == ["generic"]
