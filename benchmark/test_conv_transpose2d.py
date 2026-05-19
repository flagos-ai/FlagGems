import contextlib

import pytest
import torch

import flag_gems
from tests import accuracy_utils as utils

from . import base


@contextlib.contextmanager
def _temporary_backend_attr(backend, name, value):
    if backend is None or not hasattr(backend, name):
        yield
        return

    old_value = getattr(backend, name)
    setattr(backend, name, value)
    try:
        yield
    finally:
        setattr(backend, name, old_value)


@contextlib.contextmanager
def _benchmark_backend_flags():
    if flag_gems.vendor_name != "nvidia":
        yield
        return

    with contextlib.ExitStack() as stack:
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "flags"):
            stack.enter_context(cudnn_backend.flags(enabled=False, allow_tf32=False))
        else:
            stack.enter_context(
                _temporary_backend_attr(cudnn_backend, "enabled", False)
            )
            stack.enter_context(
                _temporary_backend_attr(cudnn_backend, "allow_tf32", False)
            )

        cuda_backend = getattr(torch.backends, "cuda", None)
        cuda_matmul_backend = getattr(cuda_backend, "matmul", None)
        stack.enter_context(
            _temporary_backend_attr(cuda_matmul_backend, "allow_tf32", False)
        )
        yield


class ConvTranspose2DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


class ConvTranspose2DBackwardBenchmark(ConvTranspose2DBenchmark):
    pass


def _input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        with_bias,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)
    bias = torch.randn(size=(out_c,), device=device, dtype=dtype) if with_bias else None

    yield inp, weight, bias, {
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation,
    },


def _backward_input_fn(shape, dtype, device):
    for inp, weight, bias, params in _input_fn(shape, dtype, device):
        yield inp, {
            "weight": weight,
            "bias": bias,
            **params,
        }


@pytest.mark.conv_transpose2d
def test_conv_transpose2d():
    with _benchmark_backend_flags():
        bench = ConvTranspose2DBenchmark(
            input_fn=_input_fn,
            op_name="conv_transpose2d",
            torch_op=torch.nn.functional.conv_transpose2d,
            dtypes=utils.FLOAT_DTYPES,
        )
        bench.set_gems(flag_gems.conv_transpose2d)

        bench.run()


@pytest.mark.conv_transpose2d_backward
def test_conv_transpose2d_backward():
    with _benchmark_backend_flags():
        bench = ConvTranspose2DBackwardBenchmark(
            input_fn=_backward_input_fn,
            op_name="conv_transpose2d_backward",
            torch_op=torch.nn.functional.conv_transpose2d,
            dtypes=utils.FLOAT_DTYPES,
            is_backward=True,
        )
        bench.set_gems(flag_gems.conv_transpose2d)

        bench.run()
