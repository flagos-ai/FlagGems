"""
Accuracy tests for Medium-difficulty operators:
  upsample_nearest2d, scatter_reduce, median, smooth_l1_loss,
  pixel_shuffle, conv_transpose2d, avg_pool3d, max_pool3d

Coverage:
  - Small / regular / large shapes
  - float16, float32, bfloat16 dtypes
  - Boundary parameter values
  - Edge cases (empty, single-element, stride/padding combos)
"""

import pytest
import torch
import torch.nn.functional as F

import flag_gems

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ops.upsample_nearest2d import upsample_nearest2d
from ops.scatter_reduce import scatter_reduce
from ops.median import median
from ops.smooth_l1_loss import smooth_l1_loss, _NONE, _MEAN, _SUM
from ops.pixel_shuffle import pixel_shuffle
from ops.conv_transpose2d import conv_transpose2d
from ops.avg_pool3d import avg_pool3d
from ops.max_pool3d import max_pool3d

DEVICE = flag_gems.device
FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _close(a, b, dtype, equal_nan=False):
    a32, b32 = a.float(), b.float()
    atol = {torch.float16: 1e-3, torch.bfloat16: 0.016, torch.float32: 1.3e-6}.get(dtype, 1e-5)
    assert torch.allclose(a32, b32, rtol=1e-4, atol=atol, equal_nan=equal_nan), (
        f"Max diff: {(a32 - b32).abs().max().item()}"
    )


# ===========================================================================
# upsample_nearest2d
# ===========================================================================
class TestUpsampleNearest2d:
    @pytest.mark.parametrize("in_size,out_size", [
        ((1, 1), (2, 2)),
        ((8, 8), (16, 16)),
        ((32, 32), (64, 64)),
        ((64, 64), (256, 256)),
        ((7, 7), (14, 14)),
        ((3, 5), (9, 15)),
    ])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, in_size, out_size, dtype):
        n, c = 2, 3
        x = torch.randn(n, c, *in_size, dtype=dtype, device=DEVICE)
        ref = F.interpolate(x.float(), size=out_size, mode="nearest")
        res = upsample_nearest2d(x, list(out_size))
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_batch_channels(self, dtype):
        x = torch.randn(4, 16, 8, 8, dtype=dtype, device=DEVICE)
        ref = F.interpolate(x.float(), size=(32, 32), mode="nearest")
        res = upsample_nearest2d(x, [32, 32])
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_non_square(self, dtype):
        x = torch.randn(1, 1, 4, 8, dtype=dtype, device=DEVICE)
        ref = F.interpolate(x.float(), size=(8, 16), mode="nearest")
        res = upsample_nearest2d(x, [8, 16])
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large(self, dtype):
        x = torch.randn(1, 3, 64, 64, dtype=dtype, device=DEVICE)
        ref = F.interpolate(x.float(), size=(256, 256), mode="nearest")
        res = upsample_nearest2d(x, [256, 256])
        _close(res, ref, dtype)


# ===========================================================================
# scatter_reduce
# ===========================================================================
class TestScatterReduce:
    @pytest.mark.parametrize("reduce", ["sum", "prod", "amax", "amin"])
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_basic(self, reduce, dtype):
        src = torch.randn(4, 8, dtype=dtype, device=DEVICE)
        idx = torch.randint(0, 4, (4, 8), device=DEVICE)
        base = torch.zeros(4, 8, dtype=dtype, device=DEVICE)
        ref = base.clone().scatter_reduce_(0, idx, src, reduce=reduce, include_self=True)
        res = scatter_reduce(base, 0, idx, src, reduce=reduce, include_self=True)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_sum_dim1(self, dtype):
        src = torch.randn(8, 16, dtype=dtype, device=DEVICE)
        idx = torch.randint(0, 16, (8, 16), device=DEVICE)
        base = torch.zeros(8, 16, dtype=dtype, device=DEVICE)
        ref = base.clone().scatter_reduce_(1, idx, src, reduce="sum")
        res = scatter_reduce(base, 1, idx, src, reduce="sum")
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_exclude_self(self, dtype):
        src = torch.randn(4, 8, dtype=dtype, device=DEVICE)
        idx = torch.randint(0, 4, (4, 8), device=DEVICE)
        base = torch.ones(4, 8, dtype=dtype, device=DEVICE)
        ref = base.clone().scatter_reduce_(0, idx, src, reduce="sum", include_self=False)
        res = scatter_reduce(base, 0, idx, src, reduce="sum", include_self=False)
        _close(res, ref, dtype)


# ===========================================================================
# median
# ===========================================================================
class TestMedian:
    @pytest.mark.parametrize("shape,dim", [
        ((8,), 0),
        ((8, 8), 0),
        ((8, 8), 1),
        ((4, 8, 16), 1),
        ((2, 3, 4, 5), 2),
    ])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, dim, dtype):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref_v, ref_i = torch.median(x, dim=dim)
        res_v, res_i = median(x, dim=dim)
        _close(res_v, ref_v, dtype)
        assert torch.equal(res_i, ref_i)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdim(self, dtype):
        x = torch.randn(8, 16, dtype=dtype, device=DEVICE)
        ref_v, ref_i = torch.median(x, dim=1, keepdim=True)
        res_v, res_i = median(x, dim=1, keepdim=True)
        assert res_v.shape == ref_v.shape
        _close(res_v, ref_v, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_odd_even_length(self, dtype):
        for n in [7, 8]:
            x = torch.randn(n, dtype=dtype, device=DEVICE)
            ref_v, ref_i = torch.median(x, dim=0)
            res_v, res_i = median(x, dim=0)
            _close(res_v, ref_v, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large(self, dtype):
        x = torch.randn(64, 1024, dtype=dtype, device=DEVICE)
        ref_v, ref_i = torch.median(x, dim=1)
        res_v, res_i = median(x, dim=1)
        _close(res_v, ref_v, dtype)


# ===========================================================================
# smooth_l1_loss
# ===========================================================================
class TestSmoothL1Loss:
    @pytest.mark.parametrize("shape", [(64,), (32, 32), (8, 8, 8)])
    @pytest.mark.parametrize("reduction", [_NONE, _MEAN, _SUM])
    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, reduction, beta, dtype):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        y = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref = F.smooth_l1_loss(x.float(), y.float(), reduction={_NONE: "none", _MEAN: "mean", _SUM: "sum"}[reduction], beta=beta)
        res = smooth_l1_loss(x, y, reduction=reduction, beta=beta)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero_diff(self, dtype):
        x = torch.randn(64, dtype=dtype, device=DEVICE)
        res = smooth_l1_loss(x, x, reduction=_MEAN)
        assert res.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large(self, dtype):
        x = torch.randn(1024, 1024, dtype=dtype, device=DEVICE)
        y = torch.randn(1024, 1024, dtype=dtype, device=DEVICE)
        ref = F.smooth_l1_loss(x.float(), y.float(), reduction="mean")
        res = smooth_l1_loss(x, y, reduction=_MEAN)
        _close(res, ref, dtype)


# ===========================================================================
# pixel_shuffle
# ===========================================================================
class TestPixelShuffle:
    @pytest.mark.parametrize("n,c,h,w,r", [
        (1, 4, 2, 2, 2),
        (2, 9, 4, 4, 3),
        (1, 16, 8, 8, 4),
        (4, 4, 32, 32, 2),
        (1, 1, 1, 1, 1),
    ])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, n, c, h, w, r, dtype):
        x = torch.randn(n, c * r * r, h, w, dtype=dtype, device=DEVICE)
        ref = F.pixel_shuffle(x.float(), r)
        res = pixel_shuffle(x, r)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large(self, dtype):
        x = torch.randn(2, 4, 64, 64, dtype=dtype, device=DEVICE)
        ref = F.pixel_shuffle(x.float(), 2)
        res = pixel_shuffle(x, 2)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_output_shape(self, dtype):
        r = 3
        x = torch.randn(1, 9, 4, 4, dtype=dtype, device=DEVICE)
        res = pixel_shuffle(x, r)
        assert res.shape == (1, 1, 12, 12)


# ===========================================================================
# conv_transpose2d
# ===========================================================================
class TestConvTranspose2d:
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("stride,padding,output_padding", [
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 1),
        (1, 1, 0),
    ])
    def test_forward(self, dtype, stride, padding, output_padding):
        x = torch.randn(2, 4, 8, 8, dtype=dtype, device=DEVICE)
        w = torch.randn(4, 2, 3, 3, dtype=dtype, device=DEVICE)
        ref = F.conv_transpose2d(x, w, stride=stride, padding=padding, output_padding=output_padding)
        res = conv_transpose2d(x, w, stride=stride, padding=padding, output_padding=output_padding)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_with_bias(self, dtype):
        x = torch.randn(2, 4, 8, 8, dtype=dtype, device=DEVICE)
        w = torch.randn(4, 2, 3, 3, dtype=dtype, device=DEVICE)
        b = torch.randn(2, dtype=dtype, device=DEVICE)
        ref = F.conv_transpose2d(x, w, bias=b)
        res = conv_transpose2d(x, w, bias=b)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_groups(self, dtype):
        x = torch.randn(2, 4, 8, 8, dtype=dtype, device=DEVICE)
        w = torch.randn(4, 1, 3, 3, dtype=dtype, device=DEVICE)
        ref = F.conv_transpose2d(x, w, groups=4)
        res = conv_transpose2d(x, w, groups=4)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_dilation(self, dtype):
        x = torch.randn(1, 2, 16, 16, dtype=dtype, device=DEVICE)
        w = torch.randn(2, 2, 3, 3, dtype=dtype, device=DEVICE)
        ref = F.conv_transpose2d(x, w, dilation=2)
        res = conv_transpose2d(x, w, dilation=2)
        _close(res, ref, dtype)


# ===========================================================================
# avg_pool3d
# ===========================================================================
class TestAvgPool3d:
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("kernel,stride,padding", [
        (2, 2, 0),
        (3, 1, 1),
        (2, 1, 0),
        (3, 2, 1),
    ])
    def test_forward(self, dtype, kernel, stride, padding):
        x = torch.randn(2, 4, 8, 8, 8, dtype=dtype, device=DEVICE)
        ref = F.avg_pool3d(x, kernel_size=kernel, stride=stride, padding=padding)
        res = avg_pool3d(x, kernel_size=kernel, stride=stride, padding=padding)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_count_include_pad(self, dtype):
        x = torch.randn(1, 1, 4, 4, 4, dtype=dtype, device=DEVICE)
        ref = F.avg_pool3d(x, 2, padding=1, count_include_pad=False)
        res = avg_pool3d(x, 2, padding=1, count_include_pad=False)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_large(self, dtype):
        x = torch.randn(2, 8, 16, 16, 16, dtype=dtype, device=DEVICE)
        ref = F.avg_pool3d(x, 2, stride=2)
        res = avg_pool3d(x, 2, stride=2)
        _close(res, ref, dtype)


# ===========================================================================
# max_pool3d
# ===========================================================================
class TestMaxPool3d:
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("kernel,stride,padding", [
        (2, 2, 0),
        (3, 1, 1),
        (2, 1, 0),
    ])
    def test_forward(self, dtype, kernel, stride, padding):
        x = torch.randn(2, 4, 8, 8, 8, dtype=dtype, device=DEVICE)
        ref = F.max_pool3d(x, kernel_size=kernel, stride=stride, padding=padding)
        res = max_pool3d(x, kernel_size=kernel, stride=stride, padding=padding)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_return_indices(self, dtype):
        x = torch.randn(1, 1, 4, 4, 4, dtype=dtype, device=DEVICE)
        ref_out, ref_idx = F.max_pool3d(x, 2, return_indices=True)
        res_out, res_idx = max_pool3d(x, 2, return_indices=True)
        _close(res_out, ref_out, dtype)
        assert torch.equal(res_idx, ref_idx)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_ceil_mode(self, dtype):
        x = torch.randn(1, 1, 5, 5, 5, dtype=dtype, device=DEVICE)
        ref = F.max_pool3d(x, 2, ceil_mode=True)
        res = max_pool3d(x, 2, ceil_mode=True)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_large(self, dtype):
        x = torch.randn(2, 8, 16, 16, 16, dtype=dtype, device=DEVICE)
        ref = F.max_pool3d(x, 2, stride=2)
        res = max_pool3d(x, 2, stride=2)
        _close(res, ref, dtype)
