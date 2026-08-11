import math

import pytest
import torch

import flag_gems

from . import base, consts

_PI = 3.1415926535897932384626433832795028841971

# torch has no polygamma on every backend (e.g. NPU), where it silently falls
# back to CPU and is unusable as a reference at benchmark shapes. There we
# compose the same math from primitives so the baseline stays on device, and
# measure the gems kernel directly via gems_op -- the harness would otherwise
# time the gems side by re-running torch_op under use_gems, which no longer
# routes through torch.polygamma here.
_DEVICE_REF = flag_gems.device not in ("cuda", "cpu")


# The composed baseline costs ~dozens of launches per call, so profiling it at
# the largest core shapes (1e9 elements) takes hours. Drop those where the
# composed reference is in use; CUDA/CPU keep the full shape list.
class _TrimHugeShapes:
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if _DEVICE_REF:
            self.shapes = [s for s in self.shapes if math.prod(s) <= 2**24]


class _UnaryBenchmark(_TrimHugeShapes, base.UnaryPointwiseBenchmark):
    pass


class _UnaryOutBenchmark(_TrimHugeShapes, base.UnaryPointwiseOutBenchmark):
    pass


def _trigamma_composed(x):
    # Same reflection + recurrence + asymptotic series the kernel uses.
    reflect = x < 0.5
    sin_pi_x = torch.sin(_PI * x)
    result = torch.where(
        reflect, -(_PI * _PI) / (sin_pi_x * sin_pi_x), torch.zeros_like(x)
    )
    y = torch.where(reflect, 1.0 - x, x)
    for _ in range(6):
        result = result + 1.0 / (y * y)
        y = y + 1.0
    iyy = 1.0 / (y * y)
    result = (
        result
        + (
            1.0
            + 1.0 / (2.0 * y)
            + iyy * (1.0 / 6.0 - iyy * (1.0 / 30.0 - iyy * (1.0 / 42.0)))
        )
        / y
    )
    return torch.where(reflect, -result, result)


def _torch_polygamma1(a, out=None):
    if not _DEVICE_REF:
        return torch.polygamma(1, a) if out is None else torch.polygamma(1, a, out=out)
    res = _trigamma_composed(a.to(torch.float32)).to(a.dtype)
    return out.copy_(res) if out is not None else res


@pytest.mark.polygamma
def test_polygamma():
    bench = _UnaryBenchmark(
        op_name="polygamma",
        torch_op=_torch_polygamma1,
        gems_op=(lambda a: flag_gems.polygamma(1, a)) if _DEVICE_REF else None,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.polygamma_
def test_polygamma_inplace():
    bench = _UnaryBenchmark(
        op_name="polygamma_",
        torch_op=lambda a: (
            a.copy_(_torch_polygamma1(a)) if _DEVICE_REF else a.polygamma_(1)
        ),
        gems_op=(lambda a: flag_gems.polygamma_(a, 1)) if _DEVICE_REF else None,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.polygamma_out
def test_polygamma_out():
    bench = _UnaryOutBenchmark(
        op_name="polygamma_out",
        torch_op=lambda a, out: _torch_polygamma1(a, out=out),
        gems_op=(
            (lambda a, out: flag_gems.polygamma_out(1, a, out)) if _DEVICE_REF else None
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
