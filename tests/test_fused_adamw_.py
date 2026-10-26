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
import itertools
from contextlib import contextmanager

import pytest
import torch

import flag_gems

from .conftest import TO_CPU

DTYPES = [
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_bf16,
            reason="Backend does not support BF16",
        ),
    ),
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name in ("ascend", "iluvatar", "mthreads")
            or not flag_gems.runtime.device.support_fp64,
            reason="Backend fused AdamW policy excludes FP64",
        ),
    ),
]
OPTIONS = list(itertools.product((False, True), (False, True), (0.0, 0.1)))
DEVICE_API = flag_gems.runtime.torch_device_fn
GRAPH_TYPE = next(
    (
        getattr(DEVICE_API, name)
        for name in ("CUDAGraph", "MUSAGraph", "NPUGraph")
        if hasattr(DEVICE_API, name)
    ),
    None,
)


def _inputs(dtype, amsgrad, layout="contiguous", mixed=False, shapes=None):
    shapes = shapes if shapes is not None else [(17,), (3, 7), (0,), ()]
    state_dtype = torch.bfloat16 if mixed else dtype
    groups = [[] for _ in range(6)]
    for i, shape in enumerate(shapes):
        p = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        g = torch.randn_like(p)
        m = torch.randn_like(p, dtype=state_dtype) * 0.1
        v = torch.rand_like(p, dtype=state_dtype) + 0.1
        mx = torch.rand_like(p, dtype=state_dtype) + 0.2
        if layout == "transpose" and p.ndim == 2:
            p, g, m, v, mx = (x.t() for x in (p, g, m, v, mx))
        elif layout == "channels_last":
            n, c, h, w = shape
            strides = (c * h * w, 1, w * c, c)
            p, g, m, v, mx = (x.as_strided(shape, strides) for x in (p, g, m, v, mx))
        for group, x in zip(groups[:4], (p, g, m, v)):
            group.append(x)
        if amsgrad:
            groups[4].append(mx)
        groups[5].append(torch.tensor(float(i + 1), device=p.device))
    return groups


def _clone(groups, device=None):
    return [[x.detach().clone().to(device or x.device) for x in xs] for xs in groups]


def _kwargs(**updates):
    values = dict(
        lr=0.003,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.1,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )
    values.update(updates)
    return values


def _reference(groups, **kw):
    """Independent CPU opmath oracle; all state stores round only at the end."""
    if kw.get("found_inf") is not None and float(kw["found_inf"].cpu()) == 1:
        return
    lr = float(kw["lr"])
    for i, p in enumerate(groups[0]):
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        g = groups[1][i].to(dtype)
        if kw.get("grad_scale") is not None:
            g = g / float(kw["grad_scale"].cpu())
            groups[1][i].copy_(g)
        if kw["maximize"]:
            g = -g
        m = groups[2][i].to(dtype) * kw["beta1"] + g * (1 - kw["beta1"])
        v = groups[3][i].to(dtype) * kw["beta2"] + g * g * (1 - kw["beta2"])
        variance = v
        if kw["amsgrad"]:
            maximum = groups[4][i].to(dtype)
            variance = torch.where(maximum < v, v, maximum)
            groups[4][i].copy_(variance)
        step = float(groups[5][i])
        bc1 = 1 - kw["beta1"] ** step
        bc2 = 1 - kw["beta2"] ** step
        denom = variance.sqrt() / bc2**0.5 + kw["eps"]
        p.copy_(p.to(dtype) * (1 - lr * kw["weight_decay"]) - (lr / bc1) * m / denom)
        groups[2][i].copy_(m)
        groups[3][i].copy_(v)


def _assert_groups(actual, expected):
    tolerances = {
        torch.float16: (2e-3, 2e-4),
        torch.bfloat16: (1.6e-2, 2e-3),
        torch.float32: (2e-5, 2e-6),
        torch.float64: (1e-10, 1e-12),
    }
    for k, (xs, ys) in enumerate(zip(actual, expected)):
        for x, y in zip(xs, ys):
            rtol, atol = (0, 0) if k == 5 else tolerances[x.dtype]
            torch.testing.assert_close(x.cpu(), y.cpu(), rtol=rtol, atol=atol)


def _check(op, inputs, kw):
    expected = _clone(inputs, "cpu")
    cpu_kw = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
    _reference(expected, **cpu_kw)
    pointers = [[x.data_ptr() for x in xs] for xs in inputs]
    assert op(*inputs, **kw) is None
    assert pointers == [[x.data_ptr() for x in xs] for xs in inputs]
    _assert_groups(inputs, expected)


@pytest.mark.fused_adamw_
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("amsgrad,maximize,weight_decay", OPTIONS)
@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
def test_fused_adamw_(dtype, amsgrad, maximize, weight_decay, layout):
    _check(
        flag_gems._fused_adamw_,
        _inputs(dtype, amsgrad, layout),
        _kwargs(amsgrad=amsgrad, maximize=maximize, weight_decay=weight_decay),
    )


@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lr_device", ["cpu", "device"])
@pytest.mark.parametrize("amsgrad,maximize,weight_decay", OPTIONS)
def test_fused_adamw__tensor_lr(dtype, lr_device, amsgrad, maximize, weight_decay):
    lr = torch.tensor(0.003, device="cpu" if lr_device == "cpu" else flag_gems.device)
    _check(
        flag_gems._fused_adamw__tensor_lr,
        _inputs(dtype, amsgrad),
        _kwargs(lr=lr, amsgrad=amsgrad, maximize=maximize, weight_decay=weight_decay),
    )


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("found", [None, 0.0, 1.0, 2.0])
@pytest.mark.parametrize("scale", [None, 4.0])
def test_fused_adamw_amp(dtype, found, scale, tensor_lr):
    inputs = _inputs(dtype, True)
    before = _clone(inputs)
    kw = _kwargs(amsgrad=True, maximize=True)
    for key, value in (("found_inf", found), ("grad_scale", scale)):
        if value is not None:
            kw[key] = torch.tensor(value, device=flag_gems.device)
    if tensor_lr:
        kw["lr"] = torch.tensor(kw["lr"], device=flag_gems.device)
    op = flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_
    _check(op, inputs, kw)
    if found == 1:
        for xs, ys in zip(inputs, before):
            for x, y in zip(xs, ys):
                assert torch.equal(x, y)


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("step,beta1,beta2", [(1, 0.0, 0.0), (100000, 0.9999, 0.99999)])
def test_fused_adamw_mixed_state(tensor_lr, amsgrad, step, beta1, beta2):
    inputs = _inputs(torch.float32, amsgrad, mixed=True)
    for s in inputs[5]:
        s.fill_(step)
    kw = _kwargs(amsgrad=amsgrad, beta1=beta1, beta2=beta2)
    if tensor_lr:
        kw["lr"] = torch.tensor(kw["lr"], device=flag_gems.device)
    _check(
        flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_,
        inputs,
        kw,
    )


@pytest.mark.fused_adamw_
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_adamw_channels_last(dtype):
    _check(
        flag_gems._fused_adamw_,
        _inputs(dtype, True, "channels_last", shapes=[(2, 3, 5, 7)]),
        _kwargs(amsgrad=True),
    )


@pytest.mark.fused_adamw_
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_adamw_many_tensors(dtype):
    _check(
        flag_gems._fused_adamw_,
        _inputs(dtype, True, shapes=[(i * 7 + 1,) for i in range(65)]),
        _kwargs(amsgrad=True),
    )


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_fused_adamw_empty(tensor_lr):
    kw = _kwargs(lr=torch.tensor(0.1) if tensor_lr else 0.1)
    op = flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_
    assert op([], [], [], [], [], [], **kw) is None
    with pytest.raises((ValueError, RuntimeError)):
        op([], [torch.empty(0, device=flag_gems.device)], [], [], [], [], **kw)


@pytest.mark.fused_adamw_
@pytest.mark.parametrize(
    "kind",
    [
        "length",
        "max_list",
        "missing_max",
        "shape",
        "stride",
        "overlap",
        "mixed",
        "step_dtype",
        "step_size",
        "scale_dtype",
        "scale_size",
    ],
)
def test_fused_adamw_invalid(kind):
    inputs = _inputs(torch.float32, False, shapes=[(3, 3)])
    kw = _kwargs()
    if kind == "length":
        inputs[1] = []
    elif kind == "max_list":
        inputs[4] = [torch.zeros_like(inputs[0][0])]
    elif kind == "missing_max":
        kw["amsgrad"] = True
    elif kind == "shape":
        inputs[1][0] = inputs[1][0].view(9)
    elif kind == "stride":
        inputs[1][0] = inputs[1][0].t()
    elif kind == "overlap":
        for group in inputs[:4]:
            group[0] = group[0][:1].expand(3, 3)
    elif kind == "mixed":
        inputs[2][0] = inputs[2][0].half()
    elif kind == "step_dtype":
        inputs[5][0] = inputs[5][0].to(torch.int64)
    elif kind == "step_size":
        inputs[5][0] = torch.ones(2, device=flag_gems.device)
    elif kind == "scale_dtype":
        kw["grad_scale"] = torch.ones((), device=flag_gems.device, dtype=torch.float16)
    elif kind == "scale_size":
        kw["grad_scale"] = torch.ones(2, device=flag_gems.device)
    before = _clone(inputs)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        flag_gems._fused_adamw_(*inputs, **kw)
    _assert_groups(inputs, before)


@pytest.mark.fused_adamw_
@pytest.mark.parametrize(
    "name,value",
    [
        ("lr", -1),
        ("beta1", 1),
        ("beta2", -0.1),
        ("eps", -1),
        ("weight_decay", -1),
        ("beta1", float("nan")),
    ],
)
def test_fused_adamw_invalid_hyperparameter(name, value):
    with pytest.raises((ValueError, RuntimeError)):
        flag_gems._fused_adamw_(
            *_inputs(torch.float32, False), **_kwargs(**{name: value})
        )


@contextmanager
def _registered_adamw():
    # Integration checks need real ATen registration; ordinary precision tests
    # call the candidate directly and keep the CPU oracle independent. Restore
    # the prior registrar as in the existing special_zeta registration tests.
    library = torch.library.Library("aten", "IMPL")
    previous_registrar = flag_gems.current_work_registrar
    try:
        flag_gems.only_enable(
            lib=library,
            include=["_fused_adamw_", "_fused_adamw__tensor_lr"],
            registrar=flag_gems.GeneralOpRegistrar,
        )
        yield
    finally:
        library._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_fused_adamw_dispatch(tensor_lr, monkeypatch):
    module = importlib.import_module("flag_gems.ops._fused_adamw_")
    original = module._fused_adamw_run
    calls = []

    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_fused_adamw_run", traced)
    inputs = _inputs(torch.float32, True)
    kw = _kwargs(amsgrad=True)
    op = torch.ops.aten._fused_adamw_.default
    if tensor_lr:
        op = torch.ops.aten._fused_adamw_.tensor_lr
        kw["lr"] = torch.tensor(0.003, device=flag_gems.device)
    with _registered_adamw():
        _check(op, inputs, kw)
    assert calls, "ATen dispatch did not reach the FlagGems AdamW kernel"


@pytest.mark.fused_adamw_
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="Native CUDA differential reference"
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_fused_adamw_native(dtype, tensor_lr):
    inputs = _inputs(dtype, True)
    kw = _kwargs(
        amsgrad=True,
        maximize=True,
        grad_scale=torch.tensor(4.0, device=flag_gems.device),
    )
    native = torch.ops.aten._fused_adamw_.default
    gems = flag_gems._fused_adamw_
    if tensor_lr:
        kw["lr"] = torch.tensor(kw["lr"], device=flag_gems.device)
        native = torch.ops.aten._fused_adamw_.tensor_lr
        gems = flag_gems._fused_adamw__tensor_lr
    if TO_CPU:
        # CPU reference mode must not call the native accelerator operator.
        # Keep the AMP and both-overload coverage with the independent oracle.
        _check(gems, inputs, kw)
    else:
        expected = _clone(inputs)
        native(*expected, **kw)
        gems(*inputs, **kw)
        _assert_groups(inputs, expected)


@pytest.mark.fused_adamw_
def test_fused_adamw_optimizer():
    p = torch.randn(257, device=flag_gems.device, requires_grad=True)
    ref = p.detach().cpu().clone().requires_grad_()
    optimizer = torch.optim.AdamW([p], lr=0.003, amsgrad=True, fused=True)
    reference = torch.optim.AdamW([ref], lr=0.003, amsgrad=True, foreach=False)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        ref.grad = p.grad.cpu()
        reference.step()
        with _registered_adamw():
            optimizer.step()
        torch.testing.assert_close(p.cpu(), ref, rtol=2e-5, atol=2e-6)
        for key in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[p][key].cpu(),
                reference.state[ref][key],
                rtol=2e-5,
                atol=2e-6,
            )


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.skipif(
    GRAPH_TYPE is None or not hasattr(DEVICE_API, "graph"),
    reason="Backend graph capture API is unavailable",
)
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_fused_adamw_graph(tensor_lr):
    inputs = _inputs(torch.float32, True, shapes=[(i + 1,) for i in range(17)])
    scale = torch.tensor(4.0, device=flag_gems.device)
    found = torch.tensor(0.0, device=flag_gems.device)
    lr = torch.tensor(0.003, device=flag_gems.device) if tensor_lr else 0.003
    kw = _kwargs(amsgrad=True, grad_scale=scale, found_inf=found, lr=lr)
    op = flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_
    stream = DEVICE_API.Stream()
    stream.wait_stream(DEVICE_API.current_stream())
    with DEVICE_API.stream(stream):
        op(*inputs, **kw)
    DEVICE_API.current_stream().wait_stream(stream)
    graph = GRAPH_TYPE()
    with DEVICE_API.graph(graph):
        op(*inputs, **kw)
    for inf, rate in ((1.0, 0.007), (0.0, 0.013), (0.0, 0.005)):
        found.fill_(inf)
        if tensor_lr:
            lr.fill_(rate)
        expected = _clone(inputs, "cpu")
        _reference(
            expected,
            **{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()},
        )
        graph.replay()
        DEVICE_API.synchronize()
        _assert_groups(inputs, expected)


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
@pytest.mark.parametrize("shapes", [[(33,)] * 256, [(2**20 + 3,)]])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_adamw_launch_boundaries(tensor_lr, shapes, dtype):
    kw = _kwargs(amsgrad=True)
    if tensor_lr:
        kw["lr"] = torch.tensor(kw["lr"], device=flag_gems.device)
    op = flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_
    _check(op, _inputs(dtype, True, shapes=shapes), kw)


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("step,beta", [(1, 0.999999), (100000, 0.99999)])
def test_fused_adamw_bias_correction(tensor_lr, dtype, step, beta):
    inputs = _inputs(dtype, True, shapes=[(257,), (17,)])
    for value in inputs[5]:
        value.fill_(step)
    kw = _kwargs(amsgrad=True, beta1=beta, beta2=beta)
    if tensor_lr:
        kw["lr"] = torch.tensor(kw["lr"], device=flag_gems.device)
    op = flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_
    _check(op, inputs, kw)


@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("dtype", [torch.int64, torch.float16])
@pytest.mark.parametrize("device", ["cpu", flag_gems.device])
def test_fused_adamw_invalid_lr_dtype(dtype, device):
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        flag_gems._fused_adamw__tensor_lr(
            *_inputs(torch.float32, False),
            **_kwargs(lr=torch.tensor(1, dtype=dtype, device=device)),
        )


@pytest.mark.fused_adamw_
@pytest.mark.parametrize(
    "invalid",
    ["param_dtype", "grad_dtype", "state_dtype", "none", "shape", "stride", "overlap"],
)
def test_fused_adamw_bulk_validation(invalid):
    inputs = _inputs(torch.float32, True, shapes=[(3, 3)] * 16)
    before = [group[0].clone() for group in inputs]
    if invalid == "param_dtype":
        inputs[0][-1] = inputs[0][-1].half()
    elif invalid == "grad_dtype":
        inputs[1][-1] = inputs[1][-1].half()
    elif invalid == "state_dtype":
        inputs[2][-1] = inputs[2][-1].half()
    elif invalid == "none":
        inputs[1][-1] = None
    elif invalid == "shape":
        inputs[1][-1] = inputs[1][-1].view(9)
    elif invalid == "stride":
        inputs[1][-1] = inputs[1][-1].t()
    else:
        inputs[0][-1] = inputs[0][-1][:1].expand(3, 3)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._fused_adamw_(*inputs, **_kwargs(amsgrad=True))
    # An invalid entry at the end must be rejected before earlier tensors update.
    for group, original in zip(inputs, before):
        assert torch.equal(group[0], original)


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.parametrize("tensor_lr", [False, True])
@pytest.mark.parametrize("count", [1, 17])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_adamw_storage_offsets(dtype, count, tensor_lr):
    # Exercise unaligned dense views and fresh addresses with identical metadata.
    # Keep the old allocation alive so a cached address cannot accidentally pass.
    allocations = []
    for offset in (1, 3):
        guards = []

        def offset_view(value):
            base = torch.full(
                (value.numel() + offset + 1,),
                7.0,
                dtype=value.dtype,
                device=value.device,
            )
            view = base[offset:-1].view(value.shape)
            view.copy_(value)
            allocations.append(base)
            guards.append((base, offset))
            return view

        inputs = _inputs(
            dtype,
            True,
            mixed=dtype == torch.float32,
            shapes=[(i + 17,) for i in range(count)],
        )
        inputs = [[offset_view(x) for x in group] for group in inputs]
        kw = _kwargs(amsgrad=True, maximize=True)
        for name, value in (("grad_scale", 4.0), ("found_inf", 0.0)):
            kw[name] = offset_view(torch.tensor(value, device=flag_gems.device))
        if tensor_lr:
            kw["lr"] = offset_view(torch.tensor(kw["lr"], device=flag_gems.device))
        op = flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_
        _check(op, inputs, kw)
        for base, start in guards:
            assert torch.equal(base[:start], torch.full_like(base[:start], 7.0))
            assert float(base[-1].cpu()) == 7.0


@pytest.mark.fused_adamw_
@pytest.mark.fused_adamw__tensor_lr
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon",
    reason="HCU integer-address alignment specialization",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_fused_adamw_alignment_classes(dtype, tensor_lr):
    # Alternate aligned and unaligned storage, including a size tail, while
    # retaining old allocations. Device scalar addresses are always unaligned.
    allocations = []
    op = flag_gems._fused_adamw__tensor_lr if tensor_lr else flag_gems._fused_adamw_
    for size, offset in (
        (64, 0),
        (64, 1),
        (64, 8),
        (65, 0),
        (4096, 0),
        (4096, 1),
        (4096, 8),
        (4097, 0),
        (131072, 0),
        (131072, 1),
        (131072, 8),
        (131073, 0),
        (131072, 0),
    ):
        guards = []

        def view(value, start):
            base = torch.full(
                (value.numel() + start + 1,),
                7.0,
                dtype=value.dtype,
                device=value.device,
            )
            result = base[start:-1].view(value.shape)
            result.copy_(value)
            allocations.append(base)
            guards.append((base, start))
            return result

        shapes = [(size,)] + [(min(size, 4096),)] * 16
        inputs = _inputs(dtype, True, mixed=dtype == torch.float32, shapes=shapes)
        inputs = [
            [view(value, 1 if index == 5 else offset) for value in group]
            for index, group in enumerate(inputs)
        ]
        kw = _kwargs(
            amsgrad=True,
            maximize=True,
            grad_scale=view(torch.tensor(4.0, device=flag_gems.device), 1),
            found_inf=view(torch.tensor(0.0, device=flag_gems.device), 1),
        )
        if tensor_lr:
            kw["lr"] = view(torch.tensor(kw["lr"], device=flag_gems.device), 1)
        _check(op, inputs, kw)
        kw["found_inf"].fill_(1.0)
        _check(op, inputs, kw)
        for base, start in guards:
            assert torch.equal(base[:start], torch.full_like(base[:start], 7.0))
            assert float(base[-1].cpu()) == 7.0

    # Capture both wide FP16 vector accesses and mixed FP32/BF16 state. The
    # ordinary graph test separately covers heterogeneous, unaligned sizes.
    if dtype in (torch.float16, torch.float32) and GRAPH_TYPE is not None:
        kw["found_inf"].zero_()
        stream = DEVICE_API.Stream()
        stream.wait_stream(DEVICE_API.current_stream())
        with DEVICE_API.stream(stream):
            op(*inputs, **kw)
        DEVICE_API.current_stream().wait_stream(stream)
        graph = GRAPH_TYPE()
        with DEVICE_API.graph(graph):
            op(*inputs, **kw)
        for inf, rate in ((1.0, 0.007), (0.0, 0.013), (0.0, 0.005)):
            kw["found_inf"].fill_(inf)
            if tensor_lr:
                kw["lr"].fill_(rate)
            expected = _clone(inputs, "cpu")
            _reference(
                expected,
                **{
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in kw.items()
                },
            )
            graph.replay()
            DEVICE_API.synchronize()
            _assert_groups(inputs, expected)
