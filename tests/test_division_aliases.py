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

from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import flag_gems
import flag_gems.ops as ops
from flag_gems.ops._division_aliases import bind_division_aliases

from . import accuracy_utils as utils


def test_division_registration_metadata():
    with (Path(__file__).resolve().parents[1] / "conf/operators.yaml").open() as f:
        metadata = {op["id"]: op for op in yaml.safe_load(f)["ops"]}
    entries = []
    names = []
    expected_names = {
        f"{packet}_{overload}"
        for packet in ("div", "div_", "divide", "divide_")
        for overload in ("scalar", "tensor", "scalar_mode", "tensor_mode")
    } | {
        "div_out",
        "true_divide_out",
        "true_divide_scalar",
        "true_divide_tensor",
        "true_divide__scalar",
        "true_divide__tensor",
    }
    for registration in flag_gems._FULL_CONFIG:
        key, entry = registration[:2]
        if key.split(".")[0] not in {
            "div",
            "div_",
            "divide",
            "divide_",
            "true_divide",
            "true_divide_",
        }:
            continue
        name = key.lstrip("_").replace(".", "_").lower()
        assert entry.__name__ == name
        assert entry.__qualname__ == name
        assert metadata[name]["for"] == [key]
        assert getattr(flag_gems, entry.__name__) is entry
        assert callable(getattr(ops, name))
        assert name in ops.__all__
        assert entry not in (flag_gems.div_mode, flag_gems.div_mode_)
        entries.append(entry)
        names.append(name)
    assert len(entries) == len(set(entries)) == 22
    assert len(names) == len(set(names)) == len(expected_names) == 22
    assert set(names) == expected_names


@pytest.mark.parametrize("specialized", [False, True])
def test_backend_binding(specialized):
    def target(label):
        return lambda *args, **kwargs: (label, args, kwargs)

    sources = {
        "div_out": "true_divide_out",
        "div_scalar": "true_divide",
        "div_tensor": "true_divide",
        "div__scalar": "true_divide_",
        "div__tensor": "true_divide_",
        "divide_tensor": "divide",
        "divide_scalar": "true_divide",
        "divide__scalar": "true_divide_",
        "divide__tensor": "true_divide_",
        "true_divide_tensor": "true_divide_tensor",
        "true_divide__tensor": "true_divide_tensor_",
        "true_divide_scalar": "true_divide",
        "true_divide__scalar": "true_divide_",
        "div_scalar_mode": "div_mode",
        "div__scalar_mode": "div_mode_",
        "div_tensor_mode": "div_mode",
        "div__tensor_mode": "div_mode_",
        "divide_scalar_mode": "div_mode",
        "divide__scalar_mode": "div_mode_",
        "divide_tensor_mode": "div_mode",
        "divide__tensor_mode": "div_mode_",
    }
    assert len(sources) == 21
    registry = {name: target(name) for name in set(sources.values())}
    bind_division_aliases(registry)
    defaults = registry.copy()
    for name in set(sources.values()):
        registry[name] = target("backend_" + name)
    if specialized:
        for name in sources:
            registry[name] = target("explicit_" + name)
    bind_division_aliases(registry, defaults)
    for name, source in sources.items():
        assert registry[name].__name__ == name
        assert registry[name].__qualname__ == name
        label = "explicit_" + name if specialized else "backend_" + source
        assert registry[name](1, out=2) == (label, (1,), {"out": 2})
        if source in ("div_mode", "div_mode_"):
            for rounding_mode in (None, "trunc", "floor"):
                assert registry[name](1, 2, rounding_mode=rounding_mode) == (
                    label,
                    (1, 2),
                    {"rounding_mode": rounding_mode},
                )
    assert len({id(registry[name]) for name in sources}) == len(sources)


def _check_registration(op_id, aten_key):
    previous = flag_gems.current_work_registrar
    library = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(
            lib=library, include=[op_id], registrar=flag_gems.GeneralOpRegistrar
        )
        assert flag_gems.current_work_registrar.get_all_keys() == [aten_key]
        assert flag_gems.current_work_registrar.get_all_ops() == [op_id]
    finally:
        try:
            library._destroy()
        finally:
            flag_gems.current_work_registrar = previous


_DISPATCH_CASES = [
    pytest.param(key, entry.__name__, mode, id=f"{key}-{mode}")
    for key, entry, *rest in flag_gems._FULL_CONFIG
    if key.split(".")[0]
    in {"div", "div_", "divide", "divide_", "true_divide", "true_divide_"}
    for mode in ([None, "trunc", "floor"] if key.endswith("_mode") else [None])
]


@pytest.mark.parametrize("aten_key,op_id,mode", _DISPATCH_CASES)
def test_division_dispatch_interception(
    monkeypatch, inplace_case, aten_key, op_id, mode
):
    """Prove ATen reaches the registered Gems entry, not a native fallback."""
    c = inplace_case
    packet, overload = aten_key.split(".")
    op = getattr(getattr(torch.ops.aten, packet), overload)
    other = -2.5 if overload.startswith("Scalar") else c.b
    ref_other = -2.5 if overload.startswith("Scalar") else c.ref_b
    kwargs = {"rounding_mode": mode} if overload.endswith("_mode") else {}
    ref_kwargs = kwargs.copy()
    if overload == "out":
        base, out, ref_base, ref_out = _make_out(c)
        kwargs["out"] = out
        ref_kwargs["out"] = ref_out
    expected = op(c.ref_a, ref_other, **ref_kwargs)
    calls = []
    registered = []
    original_impl = torch.library.Library.impl
    entry = getattr(flag_gems, op_id)

    def intercept(library, name, fn, *args, **impl_kwargs):
        if name == aten_key:
            assert fn is entry
            registered.append(name)

            @wraps(fn)
            def counted(*call_args, **call_kwargs):
                calls.append(name)
                return entry(*call_args, **call_kwargs)

            fn = counted
        return original_impl(library, name, fn, *args, **impl_kwargs)

    monkeypatch.setattr(torch.library.Library, "impl", intercept)
    previous = getattr(flag_gems, "current_work_registrar", None)
    try:
        with flag_gems.use_gems(include=[op_id]):
            assert flag_gems.current_work_registrar.get_all_keys() == [aten_key]
            assert flag_gems.current_work_registrar.get_all_ops() == [op_id]
            actual = op(c.a, other, **kwargs)
    finally:
        flag_gems.current_work_registrar = previous
    assert registered == [aten_key]
    assert calls == [aten_key]
    if overload == "out":
        _check_out(actual, out, ref_out, base, ref_base)
    elif packet.endswith("_"):
        _check_inplace(c, actual, expected)
    else:
        _check_result(actual, expected)


@pytest.fixture(params=[False, True], ids=["contiguous", "noncontiguous"])
def noncontiguous(request):
    return request.param


def _make_case(dtype, noncontiguous):
    width = 12 if noncontiguous else 6
    host = (torch.arange(4 * width, dtype=dtype) - 23).reshape(4, width)
    base = host.to(flag_gems.device)
    ref_base = base.cpu().clone()
    a = base[:, ::2] if noncontiguous else base
    ref_a = ref_base[:, ::2] if noncontiguous else ref_base
    assert a.is_contiguous() == (not noncontiguous)
    # Shape (1, 6) broadcasts over the numerator's first dimension.
    b = torch.tensor([[2, -3, 4, -5, 6, 7]], dtype=dtype, device=flag_gems.device)
    return SimpleNamespace(
        a=a,
        b=b,
        ref_a=ref_a,
        ref_b=b.cpu(),
        base=base,
        ref_base=ref_base,
        before=ref_base.clone(),
    )


@pytest.fixture(params=[torch.float32, torch.int32], ids=["float32", "int32"])
def division_case(request, noncontiguous):
    return _make_case(request.param, noncontiguous)


@pytest.fixture
def inplace_case(noncontiguous):
    # Integer true division cannot be written back to an integer input.
    return _make_case(torch.float32, noncontiguous)


@pytest.fixture(params=[3, -2.5], ids=["integer_scalar", "float_scalar"])
def scalar(request):
    return request.param


def _check_result(res_out, ref_out):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_close(res_out.cpu(), ref_out, ref_out.dtype)


def _check_inplace(case, res_out, ref_out):
    assert res_out is case.a
    assert res_out.data_ptr() == case.a.data_ptr()
    assert case.a.untyped_storage().data_ptr() == case.base.untyped_storage().data_ptr()
    _check_result(res_out, ref_out)
    # Check the shared base, including untouched columns of a strided view.
    _check_result(case.base, case.ref_base)
    if not case.a.is_contiguous():
        assert torch.equal(case.base.cpu()[:, 1::2], case.before[:, 1::2])
    assert not torch.equal(case.base.cpu(), case.before)


def _make_out(case):
    dtype = case.a.dtype if case.a.is_floating_point() else torch.get_default_dtype()
    base = torch.full((4, 12), 123.0, dtype=dtype, device=flag_gems.device)
    ref_base = base.cpu().clone()
    out = base[:, ::2]
    ref_out = ref_base[:, ::2]
    assert not out.is_contiguous()
    return base, out, ref_base, ref_out


def _check_out(res_out, out, ref_out, base, ref_base):
    assert res_out is out
    assert not res_out.is_contiguous()
    _check_result(res_out, ref_out)
    _check_result(base, ref_base)


@pytest.mark.div_out
def test_div_out(division_case):
    _check_registration("div_out", "div.out")
    c = division_case
    base, out, ref_base, ref_out = _make_out(c)
    torch.ops.aten.div.out(c.ref_a, c.ref_b, out=ref_out)
    res_out = flag_gems.div_out(c.a, c.b, out=out)
    _check_out(res_out, out, ref_out, base, ref_base)


@pytest.mark.div_scalar
def test_div_scalar(division_case, scalar):
    _check_registration("div_scalar", "div.Scalar")
    c = division_case
    ref_out = torch.ops.aten.div.Scalar(c.ref_a, scalar)
    res_out = flag_gems.div_scalar(c.a, scalar)
    _check_result(res_out, ref_out)


@pytest.mark.div__scalar
def test_div__scalar(inplace_case, scalar):
    _check_registration("div__scalar", "div_.Scalar")
    c = inplace_case
    ref_out = torch.ops.aten.div_.Scalar(c.ref_a, scalar)
    res_out = flag_gems.div__scalar(c.a, scalar)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.div_tensor
def test_div_tensor(division_case):
    _check_registration("div_tensor", "div.Tensor")
    c = division_case
    ref_out = torch.ops.aten.div.Tensor(c.ref_a, c.ref_b)
    res_out = flag_gems.div_tensor(c.a, c.b)
    _check_result(res_out, ref_out)


@pytest.mark.div__tensor
def test_div__tensor(inplace_case):
    _check_registration("div__tensor", "div_.Tensor")
    c = inplace_case
    ref_out = torch.ops.aten.div_.Tensor(c.ref_a, c.ref_b)
    res_out = flag_gems.div__tensor(c.a, c.b)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.divide_tensor
def test_divide_tensor(division_case):
    _check_registration("divide_tensor", "divide.Tensor")
    c = division_case
    ref_out = torch.ops.aten.divide.Tensor(c.ref_a, c.ref_b)
    res_out = flag_gems.divide_tensor(c.a, c.b)
    _check_result(res_out, ref_out)


@pytest.mark.divide__tensor
def test_divide__tensor(inplace_case):
    _check_registration("divide__tensor", "divide_.Tensor")
    c = inplace_case
    ref_out = torch.ops.aten.divide_.Tensor(c.ref_a, c.ref_b)
    res_out = flag_gems.divide__tensor(c.a, c.b)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.divide_scalar
def test_divide_scalar(division_case, scalar):
    _check_registration("divide_scalar", "divide.Scalar")
    c = division_case
    ref_out = torch.ops.aten.divide.Scalar(c.ref_a, scalar)
    res_out = flag_gems.divide_scalar(c.a, scalar)
    _check_result(res_out, ref_out)


@pytest.mark.divide__scalar
def test_divide__scalar(inplace_case, scalar):
    _check_registration("divide__scalar", "divide_.Scalar")
    c = inplace_case
    ref_out = torch.ops.aten.divide_.Scalar(c.ref_a, scalar)
    res_out = flag_gems.divide__scalar(c.a, scalar)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.true_divide_tensor
def test_true_divide_tensor(division_case):
    _check_registration("true_divide_tensor", "true_divide.Tensor")
    c = division_case
    ref_out = torch.ops.aten.true_divide.Tensor(c.ref_a, c.ref_b)
    res_out = flag_gems.true_divide_tensor(c.a, c.b)
    _check_result(res_out, ref_out)


@pytest.mark.true_divide__tensor
def test_true_divide__tensor(inplace_case):
    _check_registration("true_divide__tensor", "true_divide_.Tensor")
    c = inplace_case
    ref_out = torch.ops.aten.true_divide_.Tensor(c.ref_a, c.ref_b)
    res_out = flag_gems.true_divide__tensor(c.a, c.b)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.true_divide_out
def test_true_divide_out(division_case):
    _check_registration("true_divide_out", "true_divide.out")
    c = division_case
    base, out, ref_base, ref_out = _make_out(c)
    torch.ops.aten.true_divide.out(c.ref_a, c.ref_b, out=ref_out)
    res_out = flag_gems.true_divide_out(c.a, c.b, out=out)
    _check_out(res_out, out, ref_out, base, ref_base)


@pytest.mark.true_divide_scalar
def test_true_divide_scalar(division_case, scalar):
    _check_registration("true_divide_scalar", "true_divide.Scalar")
    c = division_case
    ref_out = torch.ops.aten.true_divide.Scalar(c.ref_a, scalar)
    res_out = flag_gems.true_divide_scalar(c.a, scalar)
    _check_result(res_out, ref_out)


@pytest.mark.true_divide__scalar
def test_true_divide__scalar(inplace_case, scalar):
    _check_registration("true_divide__scalar", "true_divide_.Scalar")
    c = inplace_case
    ref_out = torch.ops.aten.true_divide_.Scalar(c.ref_a, scalar)
    res_out = flag_gems.true_divide__scalar(c.a, scalar)
    _check_inplace(c, res_out, ref_out)


@pytest.fixture(params=[None, "trunc", "floor"])
def rounding_mode(request):
    return request.param


def _prepare_mode(case, op_id, aten_key, rounding_mode):
    _check_registration(op_id, aten_key)
    packet, overload = aten_key.split(".")
    inplace = packet.endswith("_")
    if inplace and rounding_mode is None and not case.a.is_floating_point():
        pytest.skip("Integer true division cannot be written back to integer input")
    ref_op = getattr(getattr(torch.ops.aten, packet), overload)
    scalar = overload == "Scalar_mode"
    # Integer scalar preserves int32 for rounded integer in-place division.
    other = -3 if scalar else case.b
    ref_other = -3 if scalar else case.ref_b
    ref_out = ref_op(case.ref_a, ref_other, rounding_mode=rounding_mode)
    return other, ref_out


@pytest.mark.div_scalar_mode
def test_div_scalar_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "div_scalar_mode", "div.Scalar_mode", rounding_mode
    )
    res_out = flag_gems.div_scalar_mode(c.a, other, rounding_mode=rounding_mode)
    _check_result(res_out, ref_out)


@pytest.mark.div__scalar_mode
def test_div__scalar_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "div__scalar_mode", "div_.Scalar_mode", rounding_mode
    )
    res_out = flag_gems.div__scalar_mode(c.a, other, rounding_mode=rounding_mode)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.div_tensor_mode
def test_div_tensor_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "div_tensor_mode", "div.Tensor_mode", rounding_mode
    )
    res_out = flag_gems.div_tensor_mode(c.a, other, rounding_mode=rounding_mode)
    _check_result(res_out, ref_out)


@pytest.mark.div__tensor_mode
def test_div__tensor_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "div__tensor_mode", "div_.Tensor_mode", rounding_mode
    )
    res_out = flag_gems.div__tensor_mode(c.a, other, rounding_mode=rounding_mode)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.divide_scalar_mode
def test_divide_scalar_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "divide_scalar_mode", "divide.Scalar_mode", rounding_mode
    )
    res_out = flag_gems.divide_scalar_mode(c.a, other, rounding_mode=rounding_mode)
    _check_result(res_out, ref_out)


@pytest.mark.divide__scalar_mode
def test_divide__scalar_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "divide__scalar_mode", "divide_.Scalar_mode", rounding_mode
    )
    res_out = flag_gems.divide__scalar_mode(c.a, other, rounding_mode=rounding_mode)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.divide_tensor_mode
def test_divide_tensor_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "divide_tensor_mode", "divide.Tensor_mode", rounding_mode
    )
    res_out = flag_gems.divide_tensor_mode(c.a, other, rounding_mode=rounding_mode)
    _check_result(res_out, ref_out)


@pytest.mark.divide__tensor_mode
def test_divide__tensor_mode(division_case, rounding_mode):
    c = division_case
    other, ref_out = _prepare_mode(
        c, "divide__tensor_mode", "divide_.Tensor_mode", rounding_mode
    )
    res_out = flag_gems.divide__tensor_mode(c.a, other, rounding_mode=rounding_mode)
    _check_inplace(c, res_out, ref_out)


@pytest.mark.parametrize("scalar_operand", [False, True])
def test_div_mode_compatibility(division_case, rounding_mode, scalar_operand):
    c = division_case
    other = -3 if scalar_operand else c.b
    ref_other = -3 if scalar_operand else c.ref_b
    ref_op = (
        torch.ops.aten.div.Scalar_mode
        if scalar_operand
        else torch.ops.aten.div.Tensor_mode
    )
    ref_out = ref_op(c.ref_a, ref_other, rounding_mode=rounding_mode)
    res_out = flag_gems.div_mode(c.a, other, rounding_mode=rounding_mode)
    _check_result(res_out, ref_out)


@pytest.mark.parametrize("scalar_operand", [False, True])
def test_div_mode_inplace_compatibility(inplace_case, rounding_mode, scalar_operand):
    c = inplace_case
    other = -3 if scalar_operand else c.b
    ref_other = -3 if scalar_operand else c.ref_b
    ref_op = (
        torch.ops.aten.div_.Scalar_mode
        if scalar_operand
        else torch.ops.aten.div_.Tensor_mode
    )
    ref_out = ref_op(c.ref_a, ref_other, rounding_mode=rounding_mode)
    res_out = flag_gems.div_mode_(c.a, other, rounding_mode=rounding_mode)
    _check_inplace(c, res_out, ref_out)
