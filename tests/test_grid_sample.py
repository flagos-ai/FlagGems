import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_cpu, to_reference

device = flag_gems.device

# ---------------------------------------------------------------------------
# grid_sampler_2d tests
# ---------------------------------------------------------------------------
_GRID_SAMPLE_MODES = [
    ("bilinear", 0),
    ("nearest", 1),
    ("bicubic", 2),
]
_GRID_SAMPLE_INTERP_BICUBIC = 2
_GRID_SAMPLE_PAD_MODES = [
    ("zeros", 0),
    ("border", 1),
    ("reflection", 2),
]
_GRID_SAMPLE_OUTPUT_MASKS = [
    [True, True],
    [True, False],
    [False, True],
    [False, False],
]


def _make_grid(N, H_out, W_out, device, dtype=torch.float32):
    """Return a uniform grid in [-1, 1]."""
    return torch.rand((N, H_out, W_out, 2), device=device, dtype=dtype) * 2.0 - 1.0


def _make_grid_3d(N, D_out, H_out, W_out, device, dtype=torch.float32):
    return (
        torch.rand((N, D_out, H_out, W_out, 3), device=device, dtype=dtype) * 2.0 - 1.0
    )


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize(
    "N, C, H_in, W_in, H_out, W_out",
    [
        # small
        (1, 1, 4, 4, 3, 3),
        # standard
        (2, 3, 16, 16, 8, 8),
        # non-square
        (2, 4, 10, 20, 7, 15),
        # more channels
        (1, 16, 32, 32, 24, 24),
        # larger
        (2, 8, 64, 64, 48, 48),
    ],
)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_grid_sampler_2d(
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners,
    interp_name,
    interp_mode,
    pad_name,
    pad_mode,
    dtype,
):
    torch.manual_seed(42)
    x = torch.randn((N, C, H_in, W_in), dtype=dtype, device=device)
    # Keep grid at float32 to avoid float16 rounding differences at pixel
    # boundaries (especially for nearest mode, where a tiny coord change can
    # select a completely different pixel).
    grid = _make_grid(N, H_out, W_out, device, dtype=torch.float32)

    ref_x = to_reference(x, True)
    ref_grid = to_reference(grid, False)

    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_x, ref_grid.to(ref_x.dtype), interp_mode, pad_mode, align_corners
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(
            x, grid, interp_mode, pad_mode, align_corners
        )

    if dtype == torch.float32:
        atol = 1e-4
    else:
        atol = 1e-2

    res_out_cmp = to_cpu(res_out, ref_out)
    ref_out_cmp = ref_out
    torch.testing.assert_close(
        res_out_cmp, ref_out_cmp, rtol=1e-4, atol=atol, equal_nan=True
    )


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_grid_sampler_2d_special_values(
    align_corners, interp_name, interp_mode, pad_name, pad_mode, dtype
):
    """Test with grid coordinates at boundaries and corners."""
    torch.manual_seed(0)
    N, C, H_in, W_in = 1, 2, 8, 8
    x = torch.randn((N, C, H_in, W_in), dtype=dtype, device=device)

    # Cover boundaries and special values that exercise padding math.
    coords = torch.tensor(
        [float("-inf"), -1.0, -0.9999, 0.0, 0.9999, 1.0, float("inf")],
        dtype=dtype,
        device=device,
    )
    H_out = W_out = coords.numel()
    gx = coords.view(1, 1, W_out).expand(N, H_out, W_out)
    gy = coords.view(1, H_out, 1).expand(N, H_out, W_out)
    grid = torch.stack([gx, gy], dim=-1)

    # CPU/CUDA differ on inf-boundary handling for this special-values suite,
    # so keep the reference on the same device.
    ref_out = torch.ops.aten.grid_sampler_2d(
        x, grid, interp_mode, pad_mode, align_corners
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(
            x, grid, interp_mode, pad_mode, align_corners
        )

    if dtype == torch.float32:
        atol = 1e-4
    else:
        atol = 1e-2

    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=atol, equal_nan=True)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
def test_grid_sampler_2d_out_of_bounds(
    align_corners, interp_name, interp_mode, pad_name, pad_mode
):
    """Test with grid values outside [-1, 1] to exercise padding modes."""
    torch.manual_seed(1)
    N, C, H_in, W_in = 1, 3, 6, 6
    H_out, W_out = 4, 4
    dtype = torch.float32
    x = torch.randn((N, C, H_in, W_in), dtype=dtype, device=device)

    # grid with values outside [-1, 1]
    grid = torch.rand((N, H_out, W_out, 2), device=device, dtype=dtype) * 4.0 - 2.0

    ref_x = to_reference(x, True)
    ref_grid = to_reference(grid, True)

    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_x, ref_grid, interp_mode, pad_mode, align_corners
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(
            x, grid, interp_mode, pad_mode, align_corners
        )

    gems_assert_close(res_out, ref_out.to(dtype), dtype, atol=1e-4)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
def test_grid_sampler_2d_nan_grid(
    align_corners, interp_name, interp_mode, pad_name, pad_mode
):
    """Use same-device reference because ATen CPU/CUDA differ on NaN-grid details."""
    torch.manual_seed(5)
    x = torch.randn((1, 3, 8, 8), dtype=torch.float32, device=device)
    grid = torch.tensor(
        [[[[float("nan"), 0.0], [0.0, float("nan")], [float("nan"), float("nan")]]]],
        dtype=torch.float32,
        device=device,
    )

    ref_out = torch.ops.aten.grid_sampler_2d(
        x, grid, interp_mode, pad_mode, align_corners
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(
            x, grid, interp_mode, pad_mode, align_corners
        )

    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize(
    "shape,expect_error",
    [
        ((1, 3, 0, 5), True),
        ((1, 3, 5, 0), True),
        ((0, 3, 5, 5), False),
    ],
)
def test_grid_sampler_2d_empty_spatial_dims(shape, expect_error):
    grid = torch.empty((shape[0], 2, 2, 2), dtype=torch.float32, device=device)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    if expect_error:
        with pytest.raises(
            RuntimeError, match="expected input to have non-empty spatial dimensions"
        ):
            torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, False)
        with flag_gems.use_gems():
            with pytest.raises(
                RuntimeError,
                match="expected input to have non-empty spatial dimensions",
            ):
                torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, False)
        return

    ref_out = torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, False)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, False)
    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
def test_grid_sampler_2d_non_contiguous(
    align_corners, interp_name, interp_mode, pad_name, pad_mode
):
    """Test with non-contiguous input tensor."""
    torch.manual_seed(2)
    N, C, H_in, W_in = 2, 4, 12, 12
    H_out, W_out = 8, 8
    dtype = torch.float32
    # Create non-contiguous tensor by slicing a larger tensor
    x_large = torch.randn((N * 2, C * 2, H_in, W_in), dtype=dtype, device=device)
    x = x_large[::2, ::2]  # non-contiguous
    grid = _make_grid(N, H_out, W_out, device, dtype)

    ref_x = to_reference(x.contiguous(), True)
    ref_grid = to_reference(grid, True)

    ref_out = torch.ops.aten.grid_sampler_2d(
        ref_x, ref_grid, interp_mode, pad_mode, align_corners
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(
            x, grid, interp_mode, pad_mode, align_corners
        )

    gems_assert_close(res_out, ref_out.to(dtype), dtype, atol=1e-4)


@pytest.mark.grid_sampler_2d
def test_grid_sampler_2d_identity_grid():
    """Identity grid should reproduce the input (bilinear, align_corners=True)."""
    torch.manual_seed(3)
    N, C, H, W = 1, 3, 8, 8
    dtype = torch.float32
    x = torch.randn((N, C, H, W), dtype=dtype, device=device)

    # Build identity grid: output[h,w] samples input[h,w]
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)

    ref_out = torch.ops.aten.grid_sampler_2d(
        to_reference(x, True), to_reference(grid, True), 0, 0, True
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, True)

    gems_assert_close(res_out, ref_out, dtype, atol=1e-4)


@pytest.mark.grid_sampler_2d
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
def test_grid_sampler_2d_large(interp_name, interp_mode, pad_name, pad_mode):
    """Large-scale test for performance-relevant shapes."""
    torch.manual_seed(4)
    N, C, H_in, W_in = 2, 64, 128, 128
    H_out, W_out = 96, 96
    dtype = torch.float32
    x = torch.randn((N, C, H_in, W_in), dtype=dtype, device=device)
    grid = _make_grid(N, H_out, W_out, device, dtype)

    ref_out = torch.ops.aten.grid_sampler_2d(
        to_reference(x, True), to_reference(grid, True), interp_mode, pad_mode, False
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler_2d(x, grid, interp_mode, pad_mode, False)

    gems_assert_close(res_out, ref_out, dtype, atol=1e-4)


@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("pad_name", ["zeros", "border", "reflection"])
def test_grid_sample_functional_2d(align_corners, interp_name, pad_name):
    torch.manual_seed(6)
    x = torch.randn((1, 3, 8, 8), dtype=torch.float32, device=device)
    grid = _make_grid(1, 6, 6, device, dtype=torch.float32)

    ref_x = to_reference(x, True)
    ref_grid = to_reference(grid, False)
    ref_out = torch.nn.functional.grid_sample(
        ref_x,
        ref_grid.to(ref_x.dtype),
        mode=interp_name,
        padding_mode=pad_name,
        align_corners=align_corners,
    ).to(torch.float32)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            x,
            grid,
            mode=interp_name,
            padding_mode=pad_name,
            align_corners=align_corners,
        )

    torch.testing.assert_close(
        to_cpu(res_out, ref_out), ref_out, rtol=1e-4, atol=1e-4, equal_nan=True
    )


@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
def test_grid_sampler_dispatch_2d(
    align_corners, interp_name, interp_mode, pad_name, pad_mode
):
    x = torch.randn((1, 3, 8, 8), dtype=torch.float32, device=device)
    grid = _make_grid(1, 6, 6, device, dtype=torch.float32)

    ref_x = to_reference(x, True)
    ref_grid = to_reference(grid, False)
    ref_out = torch.ops.aten.grid_sampler(
        ref_x, ref_grid.to(ref_x.dtype), interp_mode, pad_mode, align_corners
    ).to(torch.float32)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler(
            x, grid, interp_mode, pad_mode, align_corners
        )

    torch.testing.assert_close(
        to_cpu(res_out, ref_out), ref_out, rtol=1e-4, atol=1e-4, equal_nan=True
    )


def _run_grid_sample_autograd_case(x, grid, grad, **kwargs):
    x = x.clone().detach().requires_grad_(True)
    grid = grid.clone().detach().requires_grad_(True)
    out = torch.nn.functional.grid_sample(x, grid, **kwargs)
    out.backward(grad)
    return out.detach(), x.grad.detach(), grid.grad.detach()


def _assert_optional_tensor_close(res, ref):
    assert (res is None) == (ref is None)
    if ref is not None:
        torch.testing.assert_close(res, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
@pytest.mark.parametrize("output_mask", _GRID_SAMPLE_OUTPUT_MASKS)
def test_grid_sampler_2d_backward_direct(
    align_corners,
    interp_name,
    interp_mode,
    pad_name,
    pad_mode,
    output_mask,
):
    x = torch.randn((1, 3, 8, 8), dtype=torch.float32, device=device)
    grid = _make_grid(1, 6, 6, device, dtype=torch.float32)
    grad_output = torch.randn((1, 3, 6, 6), dtype=torch.float32, device=device)

    ref_grad_input, ref_grad_grid = torch.ops.aten.grid_sampler_2d_backward(
        grad_output, x, grid, interp_mode, pad_mode, align_corners, output_mask
    )
    with flag_gems.use_gems():
        res_grad_input, res_grad_grid = torch.ops.aten.grid_sampler_2d_backward(
            grad_output, x, grid, interp_mode, pad_mode, align_corners, output_mask
        )

    _assert_optional_tensor_close(res_grad_input, ref_grad_input)
    _assert_optional_tensor_close(res_grad_grid, ref_grad_grid)


@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("pad_name", ["zeros", "border", "reflection"])
def test_grid_sample_functional_2d_backward(align_corners, interp_name, pad_name):
    torch.manual_seed(11)
    x = torch.randn((1, 3, 8, 8), dtype=torch.float32, device=device)
    grid = _make_grid(1, 6, 6, device, dtype=torch.float32)
    kwargs = dict(mode=interp_name, padding_mode=pad_name, align_corners=align_corners)

    # Pre-compute forward output shape and generate grad before any kernel launch
    with torch.no_grad():
        dummy_out = torch.nn.functional.grid_sample(x, grid, **kwargs)
    grad = torch.randn_like(dummy_out)

    ref_out, ref_grad_x, ref_grad_grid = _run_grid_sample_autograd_case(
        x, grid, grad, **kwargs
    )
    with flag_gems.use_gems():
        res_out, res_grad_x, res_grad_grid = _run_grid_sample_autograd_case(
            x, grid, grad, **kwargs
        )

    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res_grad_x, ref_grad_x, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        res_grad_grid, ref_grad_grid, rtol=1e-4, atol=1e-4, equal_nan=True
    )


_GRID_SAMPLE_3D_MODES = [
    ("bilinear", 0),
    ("nearest", 1),
]


@pytest.mark.grid_sampler_3d
@pytest.mark.grid_sample
@pytest.mark.parametrize(
    "N, C, D_in, H_in, W_in, D_out, H_out, W_out",
    [
        (1, 1, 4, 4, 4, 3, 3, 3),
        (1, 3, 8, 8, 8, 5, 5, 5),
        (2, 4, 6, 8, 10, 4, 6, 8),
    ],
)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_3D_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_grid_sampler_3d(
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners,
    interp_name,
    interp_mode,
    pad_name,
    pad_mode,
    dtype,
):
    torch.manual_seed(7)
    x = torch.randn((N, C, D_in, H_in, W_in), dtype=dtype, device=device)
    grid = _make_grid_3d(N, D_out, H_out, W_out, device, dtype=torch.float32)

    ref_x = to_reference(x, True)
    ref_grid = to_reference(grid, False)
    ref_out = torch.nn.functional.grid_sample(
        ref_x,
        ref_grid.to(ref_x.dtype),
        mode=interp_name,
        padding_mode=pad_name,
        align_corners=align_corners,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            x,
            grid,
            mode=interp_name,
            padding_mode=pad_name,
            align_corners=align_corners,
        )

    atol = 1e-4 if dtype == torch.float32 else 1e-2
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.grid_sampler_3d
@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name", ["bilinear", "nearest"])
@pytest.mark.parametrize("pad_name", ["zeros", "border", "reflection"])
def test_grid_sampler_3d_special_values(align_corners, interp_name, pad_name):
    torch.manual_seed(8)
    x = torch.randn((1, 2, 6, 6, 6), dtype=torch.float32, device=device)
    coords = torch.tensor(
        [float("-inf"), -1.0, -0.9999, 0.0, 0.9999, 1.0, float("inf")],
        dtype=torch.float32,
        device=device,
    )
    grid_extent = coords.numel()
    gz = coords.view(1, grid_extent, 1, 1).expand(
        1, grid_extent, grid_extent, grid_extent
    )
    gy = coords.view(1, 1, grid_extent, 1).expand(
        1, grid_extent, grid_extent, grid_extent
    )
    gx = coords.view(1, 1, 1, grid_extent).expand(
        1, grid_extent, grid_extent, grid_extent
    )
    grid = torch.stack([gx, gy, gz], dim=-1)

    # CPU/CUDA differ on inf-boundary handling for this special-values suite,
    # so keep the reference on the same device.
    ref_out = torch.nn.functional.grid_sample(
        x,
        grid,
        mode=interp_name,
        padding_mode=pad_name,
        align_corners=align_corners,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            x,
            grid,
            mode=interp_name,
            padding_mode=pad_name,
            align_corners=align_corners,
        )

    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.grid_sampler_3d
@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name", ["bilinear", "nearest"])
@pytest.mark.parametrize("pad_name", ["zeros", "border", "reflection"])
def test_grid_sampler_3d_out_of_bounds(align_corners, interp_name, pad_name):
    torch.manual_seed(9)
    x = torch.randn((1, 3, 6, 6, 6), dtype=torch.float32, device=device)
    grid = torch.rand((1, 4, 4, 4, 3), device=device, dtype=torch.float32) * 4.0 - 2.0

    ref_out = torch.nn.functional.grid_sample(
        to_reference(x, True),
        to_reference(grid, True),
        mode=interp_name,
        padding_mode=pad_name,
        align_corners=align_corners,
    ).to(torch.float32)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            x,
            grid,
            mode=interp_name,
            padding_mode=pad_name,
            align_corners=align_corners,
        )

    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.grid_sampler_3d
@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name", ["bilinear", "nearest"])
@pytest.mark.parametrize("pad_name", ["zeros", "border", "reflection"])
def test_grid_sampler_3d_nan_grid(align_corners, interp_name, pad_name):
    torch.manual_seed(10)
    x = torch.randn((1, 2, 6, 6, 6), dtype=torch.float32, device=device)
    grid = torch.tensor(
        [
            [
                [
                    [
                        [float("nan"), 0.0, 0.0],
                        [0.0, float("nan"), 0.0],
                        [0.0, 0.0, float("nan")],
                        [float("nan"), float("nan"), float("nan")],
                    ]
                ]
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    ref_out = torch.nn.functional.grid_sample(
        x,
        grid,
        mode=interp_name,
        padding_mode=pad_name,
        align_corners=align_corners,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            x,
            grid,
            mode=interp_name,
            padding_mode=pad_name,
            align_corners=align_corners,
        )

    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.grid_sampler_3d
@pytest.mark.grid_sample
@pytest.mark.parametrize(
    "shape,expect_error",
    [
        ((1, 3, 0, 5, 5), True),
        ((1, 3, 5, 0, 5), True),
        ((1, 3, 5, 5, 0), True),
        ((0, 3, 5, 5, 5), False),
    ],
)
def test_grid_sampler_3d_empty_spatial_dims(shape, expect_error):
    x = torch.randn(shape, dtype=torch.float32, device=device)
    grid = torch.empty((shape[0], 2, 2, 2, 3), dtype=torch.float32, device=device)

    if expect_error:
        with pytest.raises(
            RuntimeError, match="expected input to have non-empty spatial dimensions"
        ):
            torch.nn.functional.grid_sample(
                x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        with flag_gems.use_gems():
            with pytest.raises(
                RuntimeError,
                match="expected input to have non-empty spatial dimensions",
            ):
                torch.nn.functional.grid_sample(
                    x,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
        return

    ref_out = torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.grid_sampler_3d
@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name", ["bilinear", "nearest"])
@pytest.mark.parametrize("pad_name", ["zeros", "border", "reflection"])
def test_grid_sampler_3d_non_contiguous(align_corners, interp_name, pad_name):
    torch.manual_seed(11)
    x_large = torch.randn((2, 8, 10, 12, 14), dtype=torch.float32, device=device)
    x = x_large[:, ::2, ::2, :, :]  # non-contiguous
    grid_large = _make_grid_3d(2, 10, 8, 12, device, dtype=torch.float32)
    grid = grid_large[:, ::2, :, ::2, :]  # non-contiguous

    ref_out = torch.nn.functional.grid_sample(
        to_reference(x.contiguous(), True),
        to_reference(grid.contiguous(), True),
        mode=interp_name,
        padding_mode=pad_name,
        align_corners=align_corners,
    ).to(torch.float32)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.grid_sample(
            x,
            grid,
            mode=interp_name,
            padding_mode=pad_name,
            align_corners=align_corners,
        )

    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_3D_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
def test_grid_sampler_dispatch_3d(
    align_corners, interp_name, interp_mode, pad_name, pad_mode
):
    x = torch.randn((1, 2, 6, 6, 6), dtype=torch.float32, device=device)
    grid = _make_grid_3d(1, 4, 4, 4, device, dtype=torch.float32)

    ref_x = to_reference(x, True)
    ref_grid = to_reference(grid, False)
    ref_out = torch.ops.aten.grid_sampler(
        ref_x, ref_grid.to(ref_x.dtype), interp_mode, pad_mode, align_corners
    ).to(torch.float32)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.grid_sampler(
            x, grid, interp_mode, pad_mode, align_corners
        )

    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name,interp_mode", _GRID_SAMPLE_3D_MODES)
@pytest.mark.parametrize("pad_name,pad_mode", _GRID_SAMPLE_PAD_MODES)
@pytest.mark.parametrize("output_mask", _GRID_SAMPLE_OUTPUT_MASKS)
def test_grid_sampler_3d_backward_direct(
    align_corners,
    interp_name,
    interp_mode,
    pad_name,
    pad_mode,
    output_mask,
):
    x = torch.randn((1, 2, 6, 6, 6), dtype=torch.float32, device=device)
    grid = _make_grid_3d(1, 4, 4, 4, device, dtype=torch.float32)
    grad_output = torch.randn((1, 2, 4, 4, 4), dtype=torch.float32, device=device)

    ref_grad_input, ref_grad_grid = torch.ops.aten.grid_sampler_3d_backward(
        grad_output, x, grid, interp_mode, pad_mode, align_corners, output_mask
    )
    with flag_gems.use_gems():
        res_grad_input, res_grad_grid = torch.ops.aten.grid_sampler_3d_backward(
            grad_output, x, grid, interp_mode, pad_mode, align_corners, output_mask
        )

    _assert_optional_tensor_close(res_grad_input, ref_grad_input)
    _assert_optional_tensor_close(res_grad_grid, ref_grad_grid)


@pytest.mark.grid_sample
def test_grid_sample_functional_3d_bicubic_error():
    x = torch.randn((1, 2, 4, 4, 4), dtype=torch.float32, device=device)
    grid = _make_grid_3d(1, 2, 2, 2, device, dtype=torch.float32)

    with pytest.raises(
        RuntimeError, match="bicubic interpolation only supports 4D input"
    ):
        torch.nn.functional.grid_sample(
            x, grid, mode="bicubic", padding_mode="zeros", align_corners=False
        )

    with flag_gems.use_gems():
        with pytest.raises(
            RuntimeError, match="bicubic interpolation only supports 4D input"
        ):
            torch.nn.functional.grid_sample(
                x, grid, mode="bicubic", padding_mode="zeros", align_corners=False
            )


@pytest.mark.grid_sample
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("interp_name", ["bilinear", "nearest"])
@pytest.mark.parametrize("pad_name", ["zeros", "border", "reflection"])
def test_grid_sample_functional_3d_backward(align_corners, interp_name, pad_name):
    torch.manual_seed(13)
    x = torch.randn((1, 2, 6, 6, 6), dtype=torch.float32, device=device)
    grid = _make_grid_3d(1, 4, 4, 4, device, dtype=torch.float32)
    kwargs = dict(mode=interp_name, padding_mode=pad_name, align_corners=align_corners)

    # Pre-generate grad to avoid RNG state divergence from Triton autotune
    with torch.no_grad():
        dummy_out = torch.nn.functional.grid_sample(x, grid, **kwargs)
    grad = torch.randn_like(dummy_out)

    ref_out, ref_grad_x, ref_grad_grid = _run_grid_sample_autograd_case(
        x, grid, grad, **kwargs
    )
    with flag_gems.use_gems():
        res_out, res_grad_x, res_grad_grid = _run_grid_sample_autograd_case(
            x, grid, grad, **kwargs
        )

    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res_grad_x, ref_grad_x, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        res_grad_grid, ref_grad_grid, rtol=1e-4, atol=1e-4, equal_nan=True
    )
