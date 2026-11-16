# Copyright 2026, The FlagOS Contributors.
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
#
import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)
SHAPES = [
    (7, 3),
    (3, 5, 17),
    (2, 7, 9, 11),
    (2, 3, 5, 7, 9),
    (3, 4, 2051),
    (16, 8, 128, 129),
]


def _inputs(shape, dtype, layout="contiguous", stats=True):
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if layout != "contiguous":
        physical_shape = (shape[0], *shape[2:], shape[1])
        x = torch.randn(physical_shape, dtype=dtype, device=flag_gems.device)
        x = x.permute(0, len(shape) - 1, *range(1, len(shape) - 1))
    w = torch.randn(shape[1], dtype=param_dtype, device=flag_gems.device)
    b = torch.randn_like(w)
    rm = torch.randn_like(w) if stats else None
    rv = torch.rand_like(w) + 0.5 if stats else None
    return x, w, b, rm, rv


def _reference(args, training, factor, eps):
    x, w, b, rm, rv = [
        t.detach().cpu().double() if t is not None else None for t in args
    ]
    dims = (0,) + tuple(range(2, x.ndim))
    view = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    if training:
        mean = x.mean(dims)
        var = ((x - mean.reshape(view)) ** 2).mean(dims)
        inv = torch.rsqrt(var + eps)
        if rm is not None:
            count = x.numel() // x.shape[1]
            rm = (1 - factor) * rm + factor * mean
            rv = (1 - factor) * rv + factor * var * count / (count - 1)
    else:
        mean, inv = rm, torch.rsqrt(rv + eps)
    y = (x - mean.reshape(view)) * inv.reshape(view) * w.reshape(view) + b.reshape(view)
    return (
        y,
        mean if training else torch.empty(0),
        inv if training else torch.empty(0),
        rm,
        rv,
    )


def _check(args, result, ref, training):
    dtype = args[0].dtype
    tol = (
        3e-3 if dtype == torch.float16 else (2e-5 if dtype == torch.float32 else 1e-10)
    )
    for actual, expected in zip(result[:3], ref[:3]):
        torch.testing.assert_close(
            actual.cpu().double(), expected.double(), rtol=tol, atol=tol, equal_nan=True
        )
    assert result[0].shape == args[0].shape
    assert result[0].stride() == args[0].stride()
    assert result[0].dtype == dtype
    assert result[1].dtype == args[1].dtype == result[2].dtype
    assert result[3].dtype == torch.uint8 and result[3].numel() == 0
    assert all(t.device == args[0].device for t in result)
    if args[3] is not None:
        for actual, expected in zip(args[3:], ref[3:]):
            torch.testing.assert_close(
                actual.cpu().double(), expected, rtol=tol, atol=tol, equal_nan=True
            )


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("training,stats", [(True, True), (True, False), (False, True)])
def test_cudnn_batch_norm(shape, dtype, training, stats):
    args = _inputs(shape, dtype, stats=stats)
    ref = _reference(args, training, 0.2, 1e-5)
    result = flag_gems.cudnn_batch_norm(*args, training, 0.2, 1e-5)
    _check(args, result, ref, training)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize(
    "shape,layout",
    [((2, 7, 9, 11), "channels_last"), ((2, 3, 5, 7, 9), "channels_last_3d")],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("training", [True, False])
def test_cudnn_batch_norm_layout(shape, layout, dtype, training):
    args = _inputs(shape, dtype, layout)
    ref = _reference(args, training, 0.1, 1e-5)
    _check(args, flag_gems.cudnn_batch_norm(*args, training, 0.1, 1e-5), ref, training)


@pytest.mark.cudnn_batch_norm_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    "shape,layout",
    [
        ((7, 7), "contiguous"),
        ((3, 7, 2051), "contiguous"),
        ((3, 7, 9, 11), "channels_last"),
        ((2, 7, 5, 9, 11), "channels_last_3d"),
    ],
)
def test_cudnn_batch_norm_out(dtype, training, shape, layout):
    args = _inputs(shape, dtype, layout)
    ref = _reference(args, training, 0.1, 1e-5)
    outputs = (
        torch.empty_strided(
            args[0].shape, args[0].stride(), dtype=dtype, device=flag_gems.device
        ),
        torch.empty(7 if training else 0, device=flag_gems.device, dtype=args[1].dtype),
        torch.empty(7 if training else 0, device=flag_gems.device, dtype=args[1].dtype),
        torch.empty(0, device=flag_gems.device, dtype=torch.uint8),
    )
    result = flag_gems.cudnn_batch_norm_out(
        *args,
        training,
        0.1,
        1e-5,
        **dict(zip(("out0", "out1", "out2", "out3"), outputs)),
    )
    assert all(a is b for a, b in zip(outputs[:3], result[:3]))
    assert result[3] is not outputs[3]
    _check(args, result, ref, training)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("kind", ["constant", "offset", "nan", "inf"])
def test_cudnn_batch_norm_adversarial(kind):
    args = list(_inputs((3, 4, 2051), torch.float32))
    if kind == "constant":
        args[0].fill_(13)
    elif kind == "offset":
        args[0].mul_(0.125).add_(1024)
    else:
        args[0][0, 0, 0] = float(kind)
    ref = _reference(args, True, 0.1, 1e-5)
    _check(args, flag_gems.cudnn_batch_norm(*args, True, 0.1, 1e-5), ref, True)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize(
    "bad",
    [
        "bias",
        "dtype",
        "length",
        "one_stat",
        "no_stats_eval",
        "strides",
        "rank",
        "epsilon",
    ],
)
def test_cudnn_batch_norm_invalid(bad):
    args = list(_inputs((3, 4, 17), torch.float32))
    training, eps = True, 1e-5
    if bad == "bias":
        args[2] = None
    elif bad == "dtype":
        args[1] = args[1].half()
    elif bad == "length":
        args[1] = args[1][:-1]
    elif bad == "one_stat":
        args[3] = None
    elif bad == "no_stats_eval":
        args[3:] = [None, None]
        training = False
    elif bad == "strides":
        args[0] = args[0][:, :, ::2]
    elif bad == "rank":
        args[0] = args[0].flatten()
    elif bad == "epsilon":
        eps = 0
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_batch_norm(*args, training, 0.1, eps)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("shape", [(3, 7, 17), (3, 7, 2051)])
def test_cudnn_batch_norm_autograd_after_registration(
    dtype, registered, training, shape
):
    args = _inputs(shape, dtype)
    x, w, b = [t.detach().requires_grad_() for t in args[:3]]
    xr, wr, br = [t.detach().cpu().double().requires_grad_() for t in (x, w, b)]
    rm, rv = args[3:]
    ref = torch.nn.functional.batch_norm(
        xr, rm.cpu().double(), rv.cpu().double(), wr, br, training, 0.1, 1e-5
    )
    grad = torch.randn_like(x)
    expected = torch.autograd.grad(ref, (xr, wr, br), grad.cpu().double())
    lib = torch.library.Library("aten", "IMPL") if registered else None
    try:
        if registered:
            flag_gems.only_enable(include=["cudnn_batch_norm"], lib=lib)
            y = torch.ops.aten.cudnn_batch_norm.default(
                x, w, b, rm, rv, training, 0.1, 1e-5
            )[0]
        else:
            y = flag_gems.cudnn_batch_norm(x, w, b, rm, rv, training, 0.1, 1e-5)[0]
    finally:
        if lib is not None:
            lib._destroy()
    actual = torch.autograd.grad(y, (x, w, b), grad)
    for a, e in zip(actual, expected):
        # The FP64 oracle stays on CPU in both reference modes.
        utils.gems_assert_close(
            a.cpu(), e.to(a.dtype), a.dtype, reduce_dim=math.prod(x.shape) // x.shape[1]
        )


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("training", [True, False])
def test_cudnn_batch_norm_singleton(training):
    args = _inputs((1, 3), torch.float32)
    old_mean, old_var = args[3].clone(), args[4].clone()
    result = flag_gems.cudnn_batch_norm(*args, training, 0.1, 1e-5)
    if training:
        torch.testing.assert_close(result[0], args[2].reshape(1, 3))
        torch.testing.assert_close(result[1], args[0].reshape(3))
        torch.testing.assert_close(result[2], torch.full_like(args[1], 1e-5**-0.5))
        assert torch.isnan(args[4]).all()
    else:
        torch.testing.assert_close(args[3], old_mean)
        torch.testing.assert_close(args[4], old_var)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4), (2, 3, 0)])
@pytest.mark.parametrize("training", [True, False])
def test_cudnn_batch_norm_empty(shape, training):
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_batch_norm(*_inputs(shape, torch.float32), training, 0.1, 1e-5)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("dtype", DTYPES)
def test_cudnn_batch_norm_parameter_shape(dtype):
    args = list(_inputs((3, 7, 17), dtype))
    args[1:] = [t.reshape(1, 7) for t in args[1:]]
    ref = _reference(args, True, 0.1, 1e-5)
    result = flag_gems.cudnn_batch_norm(*args, True, 0.1, 1e-5)
    _check(args, result, ref, True)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("m", [2048, 2049])
def test_cudnn_batch_norm_fused_boundary(m):
    args = _inputs((1, 3, m), torch.float32)
    ref = _reference(args, True, 0.1, 1e-5)
    _check(args, flag_gems.cudnn_batch_norm(*args, True, 0.1, 1e-5), ref, True)


@pytest.mark.cudnn_batch_norm_out
@pytest.mark.parametrize("bad", ["shape", "dtype", "overlap", "strides", "grad"])
def test_cudnn_batch_norm_out_invalid(bad):
    args = _inputs((2, 3, 17), torch.float32)
    outputs = dict(
        out0=torch.empty_like(args[0]),
        out1=torch.empty_like(args[1]),
        out2=torch.empty_like(args[1]),
        out3=torch.empty(0, dtype=torch.uint8, device=flag_gems.device),
    )
    if bad == "shape":
        outputs["out1"] = outputs["out1"][:2]
    elif bad == "dtype":
        outputs["out0"] = outputs["out0"].half()
    elif bad == "overlap":
        outputs["out0"] = args[0]
    elif bad == "strides":
        outputs["out0"] = torch.empty((2, 3, 34), device=flag_gems.device)[:, :, ::2]
    else:
        args[0].requires_grad_()
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_batch_norm_out(*args, True, 0.1, 1e-5, **outputs)


@pytest.mark.cudnn_batch_norm_out
def test_cudnn_batch_norm_out_registered():
    args = _inputs((2, 3, 17), torch.float32)
    ref = _reference(args, True, 0.1, 1e-5)
    outputs = dict(
        out0=torch.empty_like(args[0]),
        out1=torch.empty_like(args[1]),
        out2=torch.empty_like(args[1]),
        out3=torch.empty(0, dtype=torch.uint8, device=flag_gems.device),
    )
    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(include=["cudnn_batch_norm_out"], lib=lib)
        result = torch.ops.aten.cudnn_batch_norm.out(*args, True, 0.1, 1e-5, **outputs)
    finally:
        lib._destroy()
    _check(args, result, ref, True)


@pytest.mark.cudnn_batch_norm
@pytest.mark.parametrize("spatial", [4, 2051])
@pytest.mark.parametrize("training", [True, False])
def test_cudnn_batch_norm_singleton_stride(spatial, training):
    args = list(_inputs((2, 3, spatial, 1), torch.float32))
    args[0] = args[0].as_strided(args[0].shape, (3 * spatial, spatial, 1, 17))
    assert args[0].is_contiguous()
    ref = _reference(args, training, 0.1, 1e-5)
    _check(args, flag_gems.cudnn_batch_norm(*args, training, 0.1, 1e-5), ref, training)


@pytest.mark.cudnn_batch_norm_out
def test_cudnn_batch_norm_out_inference_placeholders():
    args = _inputs((2, 3, 17), torch.float32)
    out0 = torch.empty_like(args[0])
    out1, out2 = args[1].clone(), args[2].clone()
    result = flag_gems.cudnn_batch_norm_out(
        *args,
        False,
        0.1,
        1e-5,
        out0=out0,
        out1=out1,
        out2=out2,
        out3=torch.empty(0, device=flag_gems.device, dtype=torch.uint8),
    )
    assert result[1] is out1 and result[2] is out2
    torch.testing.assert_close(out1, args[1])
    torch.testing.assert_close(out2, args[2])
    torch.testing.assert_close(
        result[0].cpu().double(),
        _reference(args, False, 0.1, 1e-5)[0],
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.cudnn_batch_norm
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="NVIDIA fused and split statistics"
)
@pytest.mark.parametrize(
    "shape",
    [(1, 3, m) for m in [2048, 2049, 32768, 32769, 65568, 131072, 131073]]
    + [(16, 16, 1024), (16, 16, 4098)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cudnn_batch_norm_fast_path_boundaries(shape, dtype):
    args = list(_inputs(shape, dtype))
    args[0].mul_(0.125).add_(1024)
    # Repeated calls also catch duplicate updates from output CTAs.
    for _ in range(2):
        ref = _reference(args, True, 0.1, 1e-5)
        _check(args, flag_gems.cudnn_batch_norm(*args, True, 0.1, 1e-5), ref, True)


@pytest.mark.cudnn_batch_norm_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="NVIDIA training fast path"
)
@pytest.mark.parametrize("shape", [(16, 16, 1024), (16, 16, 4098)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cudnn_batch_norm_out_fast_path(shape, dtype):
    args = _inputs(shape, dtype)
    ref = _reference(args, True, 0.1, 1e-5)
    result = flag_gems.cudnn_batch_norm_out(
        *args,
        True,
        0.1,
        1e-5,
        out0=torch.empty_like(args[0]),
        out1=torch.empty_like(args[1]),
        out2=torch.empty_like(args[1]),
        out3=torch.empty(0, device=flag_gems.device, dtype=torch.uint8),
    )
    _check(args, result, ref, True)
