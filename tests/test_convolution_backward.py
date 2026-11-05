import itertools

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_fp64,
            reason="The target accelerator does not provide the required FP64 arithmetic",
        ),
    ),
]
MASKS = list(itertools.product((False, True), repeat=3))


def _inputs(dim, transposed, groups, dtype, layout="contiguous", batch=2):
    ci, co = 4, (8 if groups == 4 else 6)
    sizes = (9, 7, 5)[:dim]
    kernel = (3,) * dim
    stride = (2, 1, 2)[:dim]
    padding = (1,) * dim
    dilation = (1, 2, 1)[:dim]
    output_padding = (1,) * dim if transposed else (0,) * dim
    ws = (ci, co // groups, *kernel) if transposed else (co, ci // groups, *kernel)
    x = torch.randn((batch, ci, *sizes), device=flag_gems.device, dtype=dtype)
    w = torch.randn(ws, device=flag_gems.device, dtype=dtype)
    out_sizes = tuple(
        (
            (i - 1) * s - 2 * p + d * (k - 1) + op + 1
            if transposed
            else (i + 2 * p - d * (k - 1) - 1) // s + 1
        )
        for i, k, s, p, d, op in zip(
            sizes, kernel, stride, padding, dilation, output_padding
        )
    )
    go = torch.randn((batch, co, *out_sizes), device=flag_gems.device, dtype=dtype)
    if layout == "channels_last":
        order = (0, *range(2, dim + 2), 1)
        inverse = (0, dim + 1, *range(1, dim + 1))
        x, w, go = (t.permute(order).contiguous().permute(inverse) for t in (x, w, go))
    elif layout == "strided":
        x, w, go = (
            t.transpose(-1, -2).contiguous().transpose(-1, -2) for t in (x, w, go)
        )
    return go, x, w, [co], stride, padding, dilation, transposed, output_padding, groups


def _reference(args):
    # CPU FP64 is independent of vendor native coverage and precision modes.
    go, x, w, *params = args
    return torch.ops.aten.convolution_backward.default(
        go.cpu().to(torch.float64),
        x.cpu().to(torch.float64),
        w.cpu().to(torch.float64),
        *params,
        (True, True, True),
    )


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask", MASKS)
def test_convolution_backward(dim, transposed, groups, dtype, mask):
    args = _inputs(dim, transposed, groups, dtype)
    expected = _reference(args) if any(mask) else (None, None, None)
    result = flag_gems.convolution_backward(*args, mask)
    for requested, actual, reference in zip(mask, result, expected):
        if requested:
            gems_assert_close(actual.cpu(), reference, dtype)
        else:
            assert actual is None


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("layout", ["channels_last", "strided"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_convolution_backward_layout(dim, transposed, layout, dtype):
    args = _inputs(dim, transposed, 2, dtype, layout)
    result = flag_gems.convolution_backward(*args, (True, True, True))
    for actual, reference in zip(result, _reference(args)):
        gems_assert_close(actual.cpu(), reference, dtype)
    if layout == "channels_last":
        fmt = torch.channels_last if dim == 2 else torch.channels_last_3d
        assert result[0].is_contiguous(memory_format=fmt)
        assert result[1].is_contiguous(memory_format=fmt)


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("mask", MASKS)
def test_convolution_backward_empty_batch(dim, transposed, mask):
    args = _inputs(dim, transposed, 2, torch.float32, batch=0)
    args = (*args[:3], None, *args[4:])
    outputs = flag_gems.convolution_backward(*args, mask)
    shapes = [args[1].shape, args[2].shape, (6,)]
    for requested, actual, shape in zip(mask, outputs, shapes):
        if requested:
            assert actual.shape == shape
            assert torch.count_nonzero(actual).item() == 0
        else:
            assert actual is None


@pytest.mark.convolution_backward
def test_convolution_backward_empty_channel():
    x = torch.empty((2, 0, 5, 5), device=flag_gems.device)
    w = torch.empty((6, 0, 3, 3), device=flag_gems.device)
    go = torch.empty((2, 0, 3, 3), device=flag_gems.device)
    dx, dw, db = flag_gems.convolution_backward(
        go, x, w, None, [1], [0], [1], False, [0], 1, [True] * 3
    )
    assert dx.shape == x.shape and dw.shape == w.shape and db.shape == (6,)
    assert torch.count_nonzero(db).item() == 0


@pytest.mark.convolution_backward
@pytest.mark.parametrize(
    "index,value",
    [
        (4, [0]),
        (4, [1, 1, 1, 1]),
        (5, [-1]),
        (6, [0]),
        (8, [1]),
        (9, 0),
        (9, 3),
        (3, [99]),
    ],
)
def test_convolution_backward_invalid(index, value):
    args = list(_inputs(2, False, 1, torch.float32))
    args[index] = value
    with pytest.raises(RuntimeError):
        flag_gems.convolution_backward(*args, (False, False, False))


@pytest.mark.convolution_backward
def test_convolution_backward_deterministic():
    args = _inputs(3, True, 1, torch.float32)
    old = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = flag_gems.convolution_backward(*args, (True, True, True))
        second = flag_gems.convolution_backward(*args, (True, True, True))
        for a, b in zip(first, second):
            assert torch.equal(a, b)
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn_only)


@pytest.mark.convolution_backward_out
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_convolution_backward_out(dim, transposed, mask, dtype):
    args = _inputs(dim, transposed, 2, dtype)
    outputs = [
        torch.full((1,), 42, dtype=dtype, device=flag_gems.device) for _ in range(3)
    ]
    if not all(mask):
        with pytest.raises(RuntimeError, match="undefined"):
            flag_gems.convolution_backward_out(
                *args, mask, out0=outputs[0], out1=outputs[1], out2=outputs[2]
            )
        return
    expected = _reference(args)
    result = flag_gems.convolution_backward_out(
        *args, mask, out0=outputs[0], out1=outputs[1], out2=outputs[2]
    )
    for requested, out, actual, reference in zip(mask, outputs, result, expected):
        assert actual is out
        if requested:
            gems_assert_close(actual.cpu(), reference, dtype)
        else:
            assert actual.shape == (1,) and actual.item() == 42


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "batch,ci,co,spatial,groups",
    [
        (4, 16, 32, (128,), 1),
        (4, 16, 32, (32, 32), 1),
        (2, 8, 16, (12, 12, 12), 1),
        (4, 16, 32, (32, 32), 2),
        (4, 16, 32, (32, 32), 16),
        (2, 3, 5, (17, 19), 1),
    ],
)
def test_convolution_backward_large_reduction(
    dtype, transposed, batch, ci, co, spatial, groups
):
    dim = len(spatial)
    x = torch.randn((batch, ci, *spatial), device=flag_gems.device, dtype=dtype)
    shape = (ci, co // groups) if transposed else (co, ci // groups)
    w = torch.randn((*shape, *((3,) * dim)), device=flag_gems.device, dtype=dtype)
    go = torch.randn((batch, co, *spatial), device=flag_gems.device, dtype=dtype)
    args = (
        go,
        x,
        w,
        [co],
        [1] * dim,
        [1] * dim,
        [1] * dim,
        transposed,
        [0] * dim,
        groups,
    )
    expected = _reference(args)
    for mask in ([True] * 3, [False, True, False]):
        results = flag_gems.convolution_backward(*args, mask)
        for requested, actual, reference in zip(mask, results, expected):
            if requested:
                gems_assert_close(actual.cpu(), reference, dtype)
            else:
                assert actual is None


@pytest.mark.convolution_backward
@pytest.mark.skipif(
    not flag_gems.runtime.device.support_fp64,
    reason="FP64 arithmetic is unavailable",
)
def test_convolution_backward_fp64_precision():
    delta = 2.0**-40
    x = torch.tensor(
        [[[1.0 + delta, 1.0 + delta]]], dtype=torch.float64, device=flag_gems.device
    )
    w = torch.tensor([[[1.0 + delta]]], dtype=torch.float64, device=flag_gems.device)
    go = torch.ones_like(x)
    dx, dw, _ = flag_gems.convolution_backward(
        go, x, w, [1], [1], [0], [1], False, [0], 1, [True] * 3
    )
    torch.testing.assert_close(dx.cpu(), x.cpu(), rtol=0, atol=1e-14)
    torch.testing.assert_close(
        dw.cpu(),
        torch.tensor([[[2.0 + 2 * delta]]], dtype=torch.float64),
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.convolution_backward
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="Large strided allocation is validated on NVIDIA",
)
def test_convolution_backward_64bit_address():
    offset = 2**31
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 2**30:
        pytest.skip("64-bit stride check needs a 4 GiB backing allocation")
    storage = torch.empty((offset + 1,), device=flag_gems.device, dtype=torch.float16)
    storage[0] = 2
    storage[offset] = 3
    x = storage.as_strided((1, 1, 2), (0, 0, offset))
    w = torch.ones((1, 1, 1), device=flag_gems.device, dtype=x.dtype)
    go = torch.ones((1, 1, 2), device=flag_gems.device, dtype=x.dtype)
    dx, dw, db = flag_gems.convolution_backward(
        go, x, w, [1], [1], [0], [1], False, [0], 1, [True] * 3
    )
    torch.testing.assert_close(dx.cpu(), torch.ones((1, 1, 2), dtype=x.dtype))
    assert dw.item() == 5 and db.item() == 2


@pytest.mark.convolution_backward_out
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_convolution_backward_out_strided(transposed, dtype):
    args = _inputs(2, transposed, 1, dtype)
    shapes = [args[1].shape, args[2].shape, (6,)]
    outputs = [
        torch.empty((*shape, 2), device=flag_gems.device, dtype=dtype)[..., 1]
        for shape in shapes
    ]
    strides = [out.stride() for out in outputs]
    expected = _reference(args)
    for _ in range(3):
        actual = flag_gems.convolution_backward_out(
            *args, [True] * 3, out0=outputs[0], out1=outputs[1], out2=outputs[2]
        )
        for result, out, reference, steps in zip(actual, outputs, expected, strides):
            assert result is out and out.stride() == steps
            gems_assert_close(result.cpu(), reference, dtype)


@pytest.mark.convolution_backward_out
@pytest.mark.parametrize("transposed", [False, True])
def test_convolution_backward_out_input_alias(transposed):
    args = _inputs(2, transposed, 1, torch.float32)
    expected = _reference(args)
    outputs = (args[1], args[2], torch.empty((6,), device=flag_gems.device))
    result = flag_gems.convolution_backward_out(
        *args, [True] * 3, out0=outputs[0], out1=outputs[1], out2=outputs[2]
    )
    for actual, out, reference in zip(result, outputs, expected):
        assert actual is out
        gems_assert_close(actual.cpu(), reference, torch.float32)


@pytest.mark.convolution_backward
@pytest.mark.parametrize(
    "bad_kind",
    ["rank", "dtype", "device", "channels", "grad_shape", "zero_spatial", "mask"],
)
def test_convolution_backward_invalid_tensors(bad_kind):
    args = list(_inputs(2, False, 1, torch.float32))
    mask = [False] * 3
    if bad_kind == "rank":
        args[1] = args[1][0]
    elif bad_kind == "dtype":
        args[2] = args[2].to(torch.float16)
    elif bad_kind == "device":
        args[2] = args[2].cpu()
    elif bad_kind == "channels":
        args[1] = args[1][:, :3]
    elif bad_kind == "grad_shape":
        args[0] = args[0][..., :1]
    elif bad_kind == "zero_spatial":
        args[1] = args[1][..., :0]
    else:
        mask = [True, False]
    with pytest.raises(RuntimeError):
        flag_gems.convolution_backward(*args, mask)


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("size,kernel,padding", [(1, 1, 0), (3, 5, 4)])
def test_convolution_backward_spatial_boundaries(
    dim, transposed, size, kernel, padding
):
    co, ci = 5, 3
    x = torch.randn((1, ci, *((size,) * dim)), device=flag_gems.device)
    channels = (ci, co) if transposed else (co, ci)
    w = torch.randn((*channels, *((kernel,) * dim)), device=flag_gems.device)
    # For the large transposed kernel, zero padding keeps the output positive.
    padding = 0 if transposed else padding
    output_size = size + kernel - 1 if transposed else size + 2 * padding - kernel + 1
    go = torch.randn((1, co, *((output_size,) * dim)), device=flag_gems.device)
    args = (go, x, w, None, [1], [padding], [1], transposed, [0], 1)
    for actual, reference in zip(
        flag_gems.convolution_backward(*args, [True] * 3), _reference(args)
    ):
        gems_assert_close(actual.cpu(), reference, torch.float32)


@pytest.mark.convolution_backward
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("nvidia", "hygon"), reason="Launch-plan regression"
)
def test_convolution_backward_repeated_input_alias():
    x = torch.randn((2, 2, 5), device=flag_gems.device)
    w = torch.randn((2, 2, 3), device=flag_gems.device)
    params = ([2], [1], [1], [1], False, [0], 1)
    for go in (x, torch.randn_like(x), torch.randn_like(x), x):
        args = (go, x, w, *params)
        expected = _reference(args)
        actual = flag_gems.convolution_backward(*args, [True] * 3)
        for result, reference in zip(actual, expected):
            gems_assert_close(result.cpu(), reference, torch.float32)
    with pytest.raises(RuntimeError, match="invalid stride"):
        flag_gems.convolution_backward(
            x, x, w, [2], [True], [1], [1], False, [0], 1, [True] * 3
        )


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("groups", [1, 3, 33])
def test_convolution_backward_channel_tiles(dtype, transposed, groups):
    ci, co = 33, (66 if groups == 33 else 51)
    x = torch.randn((2, ci, 5, 7), device=flag_gems.device, dtype=dtype)
    channels = (ci, co // groups) if transposed else (co, ci // groups)
    w = torch.randn((*channels, 3, 3), device=flag_gems.device, dtype=dtype)
    go = torch.randn((2, co, 5, 7), device=flag_gems.device, dtype=dtype)
    args = (go, x, w, [co], [1], [1], [1], transposed, [0], groups)
    expected = _reference(args)
    for mask in ([True] * 3, [False, True, False]):
        outputs = flag_gems.convolution_backward(*args, mask)
        for requested, actual, reference in zip(mask, outputs, expected):
            if requested:
                gems_assert_close(actual.cpu(), reference, dtype)
            else:
                assert actual is None


@pytest.mark.convolution_backward
@pytest.mark.parametrize("transposed", [False, True])
def test_convolution_backward_large_dilation(transposed):
    # Large coordinate gaps must not require an equally large padded workspace.
    x = torch.randn((1, 3, 7), device=flag_gems.device)
    shape = (3, 5, 3) if transposed else (5, 3, 3)
    w = torch.randn(shape, device=flag_gems.device)
    go = torch.randn((1, 5, 7), device=flag_gems.device)
    gap = 2**20
    args = (go, x, w, [5], [1], [gap], [gap], transposed, [0], 1)
    result = flag_gems.convolution_backward(*args, [True] * 3)
    for actual, reference in zip(result, _reference(args)):
        gems_assert_close(actual.cpu(), reference, torch.float32)
