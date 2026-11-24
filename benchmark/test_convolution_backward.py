from functools import partial

import pytest
import torch

import flag_gems

from . import base


@pytest.fixture(autouse=True)
def _strict_convolution_precision():
    if flag_gems.vendor_name == "ascend":
        import torch_npu

        # Match the IEEE FP32 Gems path instead of timing native HF32 math.
        with torch_npu.npu.aclnn.flags(allow_hf32=False):
            yield
    else:
        yield


class ConvolutionBackwardBenchmark(base.GenericBenchmark):
    # batch, input channels, output channels, spatial extent, groups
    DEFAULT_SHAPES = [
        (4, 16, 32, (128,), 1),
        (4, 16, 32, (32, 32), 1),
        (2, 8, 16, (12, 12, 12), 1),
        (4, 16, 32, (32, 32), 2),
        (4, 16, 32, (32, 32), 16),
        (2, 3, 5, (17, 19), 1),
    ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.DEFAULT_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


def _inputs(shape, dtype, device, *, transposed, mask, out):
    batch, ci, co, spatial, groups = shape
    dim = len(spatial)
    k = (3,) * dim
    x = torch.randn((batch, ci, *spatial), device=device, dtype=dtype)
    ws = (ci, co // groups, *k) if transposed else (co, ci // groups, *k)
    w = torch.randn(ws, device=device, dtype=dtype)
    go = torch.randn((batch, co, *spatial), device=device, dtype=dtype)
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
        mask,
    )
    if out:
        yield (
            *args,
            {
                "out0": torch.empty_like(x),
                "out1": torch.empty_like(w),
                "out2": torch.empty((co,), device=device, dtype=dtype),
            },
        )
    else:
        yield args


DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
MASKS = [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, True),
]


def _grouped_inputs(shape, dtype, device, *, out, masks):
    for transposed in (False, True):
        for mask in masks:
            yield from _inputs(
                shape, dtype, device, transposed=transposed, mask=mask, out=out
            )


DTYPE_CASES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            dtype == torch.float64 and not flag_gems.runtime.device.support_fp64,
            reason="The target backend does not support the required FP64 arithmetic",
        ),
    )
    for dtype in DTYPES
]
OUT_MASKS = [
    pytest.param(
        mask,
        marks=pytest.mark.skipif(
            not all(mask),
            reason="ATen generated out cannot copy an undefined masked gradient into a defined Tensor",
        ),
        id=name,
    )
    for mask, name in zip(MASKS, ("dx", "dw", "db", "all"))
]


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_convolution_backward(dtype):
    bench = ConvolutionBackwardBenchmark(
        op_name="convolution_backward",
        torch_op=torch.ops.aten.convolution_backward.default,
        gems_op=flag_gems.convolution_backward,
        dtypes=[dtype],
        input_fn=partial(_grouped_inputs, out=False, masks=MASKS),
    )
    bench.run()


@pytest.mark.convolution_backward_out
@pytest.mark.parametrize("dtype", DTYPE_CASES)
@pytest.mark.parametrize("mask", OUT_MASKS)
def test_convolution_backward_out(dtype, mask):
    bench = ConvolutionBackwardBenchmark(
        op_name="convolution_backward_out",
        torch_op=torch.ops.aten.convolution_backward.out,
        gems_op=flag_gems.convolution_backward_out,
        dtypes=[dtype],
        input_fn=partial(_grouped_inputs, out=True, masks=(mask,)),
    )
    bench.run()
