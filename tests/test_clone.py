import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _clone_reference(inp, memory_format=None):
    ref = utils.to_reference(inp)
    if (
        flag_gems.vendor_name == "ascend"
        and torch.ops.aten.is_non_overlapping_and_dense(inp)
    ):
        # Device transfer can pack a dense permutation. Recreate its layout
        # before native cloning so the oracle retains preserve_format semantics.
        mirror = torch.empty_strided(
            inp.shape, inp.stride(), dtype=ref.dtype, device=ref.device
        )
        mirror.copy_(ref)
        ref = mirror
    return torch.clone(ref, memory_format=memory_format or torch.preserve_format)


@pytest.mark.clone
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clone(ref_inp)
    res_out = flag_gems.clone(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clone
@pytest.mark.parametrize(
    "case",
    [
        "contiguous",
        "expanded",
        "stepped_slice",
        "transposed",
        "channels_last",
        "permuted",
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_preserve_format_layout(dtype, case):
    torch.manual_seed(0)
    if case == "channels_last":
        cpu = torch.randn((2, 3, 4, 5), dtype=dtype).to(
            memory_format=torch.channels_last
        )
        inp = torch.empty_strided(
            cpu.shape, cpu.stride(), dtype=dtype, device=flag_gems.device
        )
        inp.copy_(cpu)
    elif case == "permuted":
        inp = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device).permute(
            2, 0, 1
        )
    else:
        base = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
        if case == "contiguous":
            inp = base
        elif case == "expanded":
            inp = base[:, :1].expand(4, 8)
        elif case == "stepped_slice":
            inp = base[:, ::2]
        else:
            inp = base.t()

    ref_out = _clone_reference(inp)
    res_out = flag_gems.clone(inp)

    assert res_out.shape == ref_out.shape
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clone
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_non_contiguous_values(dtype):
    # Cloning a non-contiguous input must produce a tensor equal to the
    # logical values of the source.
    torch.manual_seed(0)
    base = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    for inp in (base[:, ::2], base.t()):
        ref_out = torch.clone(utils.to_reference(inp))
        res_out = flag_gems.clone(inp)
        utils.gems_assert_equal(res_out, ref_out.to(res_out.dtype))


@pytest.mark.clone
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
def test_clone_autograd(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)

    res_out = flag_gems.clone(inp)
    ref_out = torch.clone(ref_inp)

    (res_in_grad,) = torch.autograd.grad(res_out.sum(), [inp])
    (ref_in_grad,) = torch.autograd.grad(ref_out.sum(), [ref_inp])

    utils.gems_assert_close(res_in_grad, ref_in_grad.to(dtype), dtype)


@pytest.mark.clone
def test_clone_autograd_non_contiguous():
    torch.manual_seed(0)
    base = torch.randn((4, 8), device=flag_gems.device, requires_grad=True)
    inp = base[:, ::2]
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)

    res_out = flag_gems.clone(inp)
    ref_out = torch.clone(ref_inp)

    (res_in_grad,) = torch.autograd.grad(res_out.sum(), [inp])
    (ref_in_grad,) = torch.autograd.grad(ref_out.sum(), [ref_inp])

    utils.gems_assert_close(res_in_grad, ref_in_grad.to(torch.float32), torch.float32)


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
        assert torch.equal(torch.autograd.grad(out.sum(), inp)[0], torch.ones_like(inp))
        state = torch.random.get_rng_state()
        assert torch.equal(state.clone(), state)
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
            assert torch.equal(out.cpu(), inp.detach().cpu())
            assert out.stride() == inp.stride()
            assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata
            (grad,) = torch.autograd.grad(out.sum(), inp)
            assert torch.equal(grad.cpu(), torch.ones(inp.shape, dtype=inp.dtype))
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
                assert torch.equal(out.cpu(), expected)
                assert out.stride() == expected.stride()
                assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata
                (grad,) = torch.autograd.grad(out.sum(), inp)
                assert torch.equal(grad.cpu(), torch.ones(inp.shape, dtype=dtype))


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
            assert torch.equal(out.cpu(), torch.full(inp.shape, value, dtype=inp.dtype))
