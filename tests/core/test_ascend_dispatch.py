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

"""Ascend dispatcher integration, separate from direct operator accuracy tests."""

import pytest
import torch

import flag_gems
from tests import accuracy_utils as utils
from tests.test_clone import _clone_reference

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend dispatcher integration"
)


@pytest.mark.clone
def test_clone_backend_only_registration():
    entry = next(item for item in flag_gems._FULL_CONFIG if item[0] == "clone")
    assert entry[1] is flag_gems.clone
    assert flag_gems.registrar._extra_dispatch_keys(entry) == ()


@pytest.mark.clone
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend dispatch regression"
)
def test_clone_cpu_and_native_autograd_unchanged(monkeypatch):
    import importlib

    generic = importlib.import_module("flag_gems.ops.clone")

    def forbid(*args, **kwargs):
        raise AssertionError("CPU clone reached the NPU copy kernel")

    def dispatch():
        return {
            row.split(":", 1)[0]: row
            for row in torch._C._dispatch_dump_table("aten::clone").splitlines()
            if row.startswith(("CPU:", "AutogradCPU:", "AutogradPrivateUse1:"))
        }

    monkeypatch.setattr(generic, "_launch_copy", forbid)
    inp = torch.randn(7, requires_grad=True)
    native_grad_type = type(inp.clone().grad_fn)
    before = dispatch()
    original = torch.clone
    with flag_gems.use_gems(include=["clone"]):
        assert torch.clone is original
        assert dispatch() == before
        out = inp.clone()
        assert type(out.grad_fn) is native_grad_type
        assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata
        utils.gems_assert_equal(
            torch.autograd.grad(out.sum(), inp)[0], torch.ones_like(inp)
        )
        state = torch.random.get_rng_state()
        utils.gems_assert_equal(state.clone(), state)
        print(
            {
                "before": before,
                "during": dispatch(),
                "grad_type": native_grad_type.__name__,
            }
        )


@pytest.mark.clone
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend dispatch regression"
)
def test_clone_registered_fake_no_launch(monkeypatch):
    import importlib

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    generic = importlib.import_module("flag_gems.ops.clone")

    def forbid(*args, **kwargs):
        raise AssertionError("Fake Tensor.clone reached the real copy kernel")

    monkeypatch.setattr(generic, "_launch_copy", forbid)
    real = torch.empty((2, 4, 4096), device=flag_gems.device, dtype=torch.bfloat16)
    mode = FakeTensorMode(shape_env=ShapeEnv())
    fake = mode.from_tensor(real, static_shapes=False)
    with flag_gems.use_gems(include=["clone"]), mode:
        for inp in (fake, fake.transpose(1, 2)):
            for fmt in (torch.preserve_format, torch.contiguous_format):
                out = inp.clone(memory_format=fmt)
                expected = torch.empty_like(inp, memory_format=fmt)
                assert out.shape == expected.shape and out.stride() == expected.stride()
                assert out.dtype == inp.dtype and out.device == inp.device
                assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata


@pytest.mark.clone
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend dispatch regression"
)
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("transposed", [False, True])
def test_clone_registered_dynamic_autograd(backend, transposed):
    with flag_gems.use_gems(include=["clone"]):
        compiled = torch.compile(
            lambda inp: inp.clone(), backend=backend, fullgraph=True, dynamic=True
        )
        for tokens in (2, 7):
            inp = torch.randn(
                (tokens, 4, 4096), dtype=torch.bfloat16, device=flag_gems.device
            )
            if transposed:
                inp = inp.transpose(1, 2)
            inp.requires_grad_()
            out = compiled(inp)
            utils.gems_assert_equal(out.cpu(), inp.detach().cpu())
            assert out.stride() == inp.stride()
            assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata
            (grad,) = torch.autograd.grad(out.sum(), inp)
            utils.gems_assert_equal(grad.cpu(), torch.ones(inp.shape, dtype=inp.dtype))
    torch._dynamo.reset()


@pytest.mark.clone
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend dispatch regression"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_clone_registered_layout_storage_gradient(dtype):
    base = torch.randn((4, 8), device=flag_gems.device, dtype=dtype)
    inputs = [base, base.t(), base[:, ::2], base[:, :1].expand(4, 8)]
    inputs += [torch.randn((), device=flag_gems.device, dtype=dtype)]
    inputs += [torch.empty((0, 4), device=flag_gems.device, dtype=dtype)]
    with flag_gems.use_gems(include=["clone"]):
        for inp in inputs:
            inp.requires_grad_()
            for fmt in (torch.preserve_format, torch.contiguous_format):
                expected = _clone_reference(inp.detach(), memory_format=fmt)
                out = inp.clone(memory_format=fmt)
                utils.gems_assert_equal(out.cpu(), expected.cpu())
                assert out.stride() == expected.stride()
                assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata
                (grad,) = torch.autograd.grad(out.sum(), inp)
                utils.gems_assert_equal(grad.cpu(), torch.ones(inp.shape, dtype=dtype))


@pytest.mark.clone
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend dispatch regression"
)
def test_clone_registered_graph_replay(monkeypatch):
    import importlib

    generic = importlib.import_module("flag_gems.ops.clone")
    original = generic._launch_copy
    launches = []

    def record_launch(*args):
        launches.append(True)
        return original(*args)

    monkeypatch.setattr(generic, "_launch_copy", record_launch)
    inp = torch.randn((7, 4, 4096), device=flag_gems.device, dtype=torch.bfloat16)
    with flag_gems.use_gems(include=["clone"]):
        inp.clone()
        torch.npu.synchronize()
        before = len(launches)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = inp.clone()
        assert len(launches) > before
        for value in (3, -2):
            inp.fill_(value)
            graph.replay()
            torch.npu.synchronize()
            utils.gems_assert_equal(
                out.cpu(), torch.full(inp.shape, value, dtype=inp.dtype)
            )


@pytest.mark.new_zeros
@pytest.mark.new_zeros_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend factories")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int64,
        torch.bool,
        torch.complex64,
    ],
)
def test_ascend_new_zeros_registered_graph_and_out(dtype, monkeypatch):
    def forbid_cuda(*args, **kwargs):
        raise AssertionError("NPU new_zeros tried to initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_cuda)
    src = torch.empty(0, dtype=dtype, device=flag_gems.device)
    fill = 11 + 3j if dtype.is_complex else 11
    parent = torch.full((3, 14), fill, dtype=dtype, device=flag_gems.device)
    expected_parent = parent.cpu()
    expected_parent[:, 1::2] = 0
    view = parent[:, 1::2]
    with flag_gems.use_gems(include=["new_zeros", "new_zeros_out"]):
        for size in ((), (0,), (17, 13)):
            result = src.new_zeros(size)
            assert result.device == src.device and result.dtype == dtype
            utils.gems_assert_equal(result.cpu(), torch.zeros(size, dtype=dtype))
        assert torch.ops.aten.new_zeros.out(src, (3, 7), out=view) is view
        utils.gems_assert_equal(parent.cpu(), expected_parent)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = src.new_zeros((17, 13))
        for _ in range(2):
            result.fill_(1)
            graph.replay()
            torch.npu.synchronize()
            utils.gems_assert_equal(result.cpu(), torch.zeros((17, 13), dtype=dtype))


@pytest.mark.new_zeros
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend factories")
def test_ascend_new_zeros_uses_source_device():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    src = torch.empty(0, dtype=torch.bfloat16, device="npu:0")
    with torch.npu.device(1), flag_gems.use_gems(include=["new_zeros"]):
        result = src.new_zeros((3, 7))
    assert result.device == src.device
    utils.gems_assert_equal(result.cpu(), torch.zeros((3, 7), dtype=src.dtype))


@pytest.mark.true_divide_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend dispatch")
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("op", ["div_", "divide_", "true_divide_"])
@pytest.mark.parametrize("overload", ["Tensor", "Scalar"])
def test_ascend_true_divide_inplace_alias(dtype, op, overload):
    cpu = torch.arange(257, dtype=torch.float32).to(dtype)
    inp = cpu.to(flag_gems.device)
    divisor = (
        torch.tensor(2, dtype=dtype, device=inp.device) if overload == "Tensor" else 2
    )
    expected = cpu.div(2)
    with flag_gems.use_gems(include=["true_divide_", "true_divide_tensor_"]):
        actual = getattr(getattr(torch.ops.aten, op), overload)(inp, divisor)
    assert actual is inp
    utils.gems_assert_equal(inp.cpu(), expected)



@pytest.mark.true_divide_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend graph replay")
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ascend_true_divide_temperature_graph_dispatch(dtype):
    cpu = (torch.arange(151936, dtype=torch.float32) % 127).reshape(1, -1).to(dtype)
    inp = cpu.to(flag_gems.device)
    temperature = torch.tensor([[2]], dtype=dtype, device=inp.device)

    def invoke():
        return inp.div_(temperature)

    with flag_gems.use_gems(include=["true_divide_", "true_divide_tensor_"]):
        invoke()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = invoke()
        for divisor in [4, 2]:
            inp.copy_(cpu)
            temperature.fill_(divisor)
            graph.replay()
            torch.npu.synchronize()
            assert actual is inp
            utils.gems_assert_equal(inp.cpu(), cpu.div(divisor))

