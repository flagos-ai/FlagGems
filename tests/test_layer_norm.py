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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    LAYER_NORM_SHAPES = [(1, 40999)]
    LAYER_NORM_AUTOGRAD_CASES = [
        ((2, 3, 512), (512,)),
    ]
    LAYER_NORM_BACKWARD_LARGE_M_SHAPES = [
        pytest.param(
            (64, 16, 64, 16),
            (16,),
            id="m65536-n16",
        ),
    ]
    LAYER_NORM_BACKWARD_STRUCTURE_SHAPES = []
    LAYER_NORM_BACKWARD_OUTPUT_MASKS = [
        pytest.param((True, True, False), id="dx-dw"),
    ]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    LAYER_NORM_SHAPES = [(200, 36), (4096, 100), (1, 40999), (100, 40499), (4096, 256)]
    LAYER_NORM_AUTOGRAD_CASES = [
        ((2, 257, 512), (512,)),
    ]
    LAYER_NORM_BACKWARD_LARGE_M_SHAPES = [
        pytest.param(
            (64, 16, 64, 16),
            (16,),
            id="m65536-n16",
        ),
        pytest.param(
            (128, 16, 16, 32),
            (32,),
            id="m32768-n32",
        ),
        pytest.param(
            (32768, 128),
            (128,),
            id="m32768-n128",
        ),
        pytest.param(
            (32768, 256),
            (256,),
            id="m32768-n256",
        ),
        pytest.param(
            (32768, 257),
            (257,),
            id="m32768-n257-tail",
        ),
    ]
    LAYER_NORM_BACKWARD_STRUCTURE_SHAPES = [
        pytest.param(
            (128, 256, 4, 8),
            (4, 8),
            id="m32768-n32-multiaxis",
        ),
        pytest.param(
            (32, 16, 32, 64),
            (32, 64),
            id="fallback-m512-n2048-multiaxis",
        ),
    ]
    LAYER_NORM_BACKWARD_OUTPUT_MASKS = [
        pytest.param((True, True, False), id="dx-dw"),
        pytest.param((True, False, True), id="dx-db"),
        pytest.param((True, False, False), id="dx-only"),
        pytest.param((False, True, True), id="affine-only"),
    ]

LAYER_NORM_BACKWARD_CASES = [(shape, shape[1:]) for shape in LAYER_NORM_SHAPES]
# Dedicated backward tests cover the kernel shapes; one case is sufficient to
# verify that autograd dispatches through the registered backward operator.
LAYER_NORM_CONTROL_DTYPES = [torch.float32]


@pytest.mark.layer_norm_backward
def test_layer_norm_backward_mthreads_vendor_dispatch():
    if flag_gems.vendor_name != "mthreads":
        pytest.skip("MThreads vendor dispatch only")

    assert flag_gems.layer_norm_backward.__module__.endswith("_mthreads.ops.layernorm")


@pytest.mark.layer_norm
@pytest.mark.parametrize("shape", LAYER_NORM_SHAPES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm(shape, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    ref_out = torch.layer_norm(
        ref_inp,
        shape[1:],
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            res_inp,
            shape[1:],
            weight=res_weight,
            bias=res_bias,
            eps=eps,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layer_norm_backward
@pytest.mark.parametrize("shape,normalized_shape", LAYER_NORM_BACKWARD_CASES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm_backward(monkeypatch, shape, normalized_shape, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        monkeypatch.setenv("DISABLE_LLVM_OPT", "1")

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    M = res_inp.numel() // math.prod(normalized_shape)
    if wb_none:
        res_weight = None
        res_bias = None
        output_mask = [True, False, False]
    else:
        res_weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
        output_mask = [True, True, True]

    _, res_mean, res_rstd = torch.ops.aten.native_layer_norm(
        res_inp,
        normalized_shape,
        res_weight,
        res_bias,
        1e-5,
    )

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_mean = utils.to_reference(res_mean, True)
    ref_rstd = utils.to_reference(res_rstd, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_layer_norm_backward(
        ref_grad,
        ref_inp,
        normalized_shape,
        ref_mean,
        ref_rstd,
        ref_weight,
        ref_bias,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_layer_norm_backward(
            res_grad,
            res_inp,
            normalized_shape,
            res_mean,
            res_rstd,
            res_weight,
            res_bias,
            output_mask,
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    if not wb_none:
        utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)


@pytest.mark.layer_norm_backward
@pytest.mark.parametrize(
    "shape,normalized_shape",
    LAYER_NORM_BACKWARD_LARGE_M_SHAPES + LAYER_NORM_BACKWARD_STRUCTURE_SHAPES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm_backward_large_m(monkeypatch, shape, normalized_shape, dtype):
    _test_layer_norm_backward_case(
        monkeypatch,
        shape,
        normalized_shape,
        (True, True, True),
        dtype,
    )


@pytest.mark.layer_norm_backward
@pytest.mark.parametrize("output_mask", LAYER_NORM_BACKWARD_OUTPUT_MASKS)
@pytest.mark.parametrize("dtype", LAYER_NORM_CONTROL_DTYPES)
def test_layer_norm_backward_output_mask(monkeypatch, output_mask, dtype):
    _test_layer_norm_backward_case(
        monkeypatch,
        (128, 16, 16, 32),
        (32,),
        output_mask,
        dtype,
    )


def _test_layer_norm_backward_case(
    monkeypatch, shape, normalized_shape, output_mask, dtype
):
    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        monkeypatch.setenv("DISABLE_LLVM_OPT", "1")

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    # output_mask controls returned gradients; dX still uses the forward weight.
    res_weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    res_bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    _, res_mean, res_rstd = torch.ops.aten.native_layer_norm(
        res_inp,
        normalized_shape,
        res_weight,
        res_bias,
        1e-5,
    )

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_mean = utils.to_reference(res_mean, True)
    ref_rstd = utils.to_reference(res_rstd, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)
    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_layer_norm_backward(
        ref_grad,
        ref_inp,
        normalized_shape,
        ref_mean,
        ref_rstd,
        ref_weight,
        ref_bias,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_layer_norm_backward(
            res_grad,
            res_inp,
            normalized_shape,
            res_mean,
            res_rstd,
            res_weight,
            res_bias,
            output_mask,
        )

    M = res_inp.numel() // math.prod(normalized_shape)
    if output_mask[0]:
        utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    else:
        assert res_in_grad is ref_in_grad is None
    if output_mask[1]:
        utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
    else:
        assert res_weight_grad is ref_weight_grad is None
    if output_mask[2]:
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)
    else:
        assert res_bias_grad is ref_bias_grad is None


@pytest.mark.layer_norm_backward
@pytest.mark.parametrize("shape,normalized_shape", LAYER_NORM_AUTOGRAD_CASES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", LAYER_NORM_CONTROL_DTYPES)
def test_layer_norm_autograd(shape, normalized_shape, dtype, wb_none):
    res_inp = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    res_grad = torch.randn_like(res_inp)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(
            normalized_shape,
            dtype=dtype,
            device=flag_gems.device,
            requires_grad=True,
        )
        res_bias = torch.randn(
            normalized_shape,
            dtype=dtype,
            device=flag_gems.device,
            requires_grad=True,
        )

    ref_inp = utils.to_reference(res_inp.detach().clone(), True).requires_grad_()
    ref_grad = utils.to_reference(res_grad, True)
    ref_weight = (
        utils.to_reference(res_weight.detach().clone(), True).requires_grad_()
        if res_weight is not None
        else None
    )
    ref_bias = (
        utils.to_reference(res_bias.detach().clone(), True).requires_grad_()
        if res_bias is not None
        else None
    )

    ref_out = torch.layer_norm(
        ref_inp,
        normalized_shape,
        weight=ref_weight,
        bias=ref_bias,
    )
    ref_out.backward(ref_grad)

    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            res_inp,
            normalized_shape,
            weight=res_weight,
            bias=res_bias,
        )
        res_out.backward(res_grad)

    M = res_inp.numel() // math.prod(normalized_shape)
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_inp.grad, ref_inp.grad, dtype)
    if not wb_none:
        utils.gems_assert_close(res_weight.grad, ref_weight.grad, dtype, reduce_dim=M)
        utils.gems_assert_close(res_bias.grad, ref_bias.grad, dtype, reduce_dim=M)
