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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.batch_norm_no_update
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_no_update(shape, dtype, affine):
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)

    ref_out = torch.nn.functional.batch_norm(
        ref_inp,
        ref_running_mean,
        ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        training=False,
        eps=eps,
    )

    (
        res_out,
        res_save_mean,
        res_save_var,
        res_reserved,
    ) = flag_gems._batch_norm_no_update(
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        0.1,
        eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


def _ref_batch_norm(inp, weight, bias, running_mean, running_var, eps):
    return torch.nn.functional.batch_norm(
        inp,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
        eps=eps,
    )


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="empty-input handling is specific to the Ascend kernel",
)
@pytest.mark.batch_norm_no_update
@pytest.mark.parametrize("shape", [(0, 3, 4, 4), (0, 8), (2, 3, 0, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_no_update_empty(shape, dtype):
    from flag_gems.runtime.backend._ascend.ops import batch_norm_no_update as ascend_bn

    # ATen BatchNorm accepts an empty batch; the spatial_dim division used to
    # raise ZeroDivisionError before the kernel was reached.
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )
    eps = 1e-5

    ref_out = _ref_batch_norm(
        utils.to_reference(inp, True),
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        utils.to_reference(running_mean, True),
        utils.to_reference(running_var, True),
        eps,
    )

    # Calls the Ascend kernel directly: the aten dispatch for an empty input
    # is handled by the generic flag_gems.ops implementation, which has its own
    # pre-existing ZeroDivisionError in batch_norm_heur_block_n.
    res_out, _, _, _ = ascend_bn(inp, weight, bias, running_mean, running_var, 0.1, eps)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="dense-offset handling is specific to the Ascend kernel",
)
@pytest.mark.batch_norm_no_update
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_no_update_non_contiguous(dtype):
    from flag_gems.runtime.backend._ascend.ops import batch_norm_no_update as ascend_bn

    # The kernels index dense NCHW storage, so a sliced input used to be read
    # with the wrong addresses (max error 4.46 before the fix).
    base = torch.randn(size=(2, 3, 8, 4), dtype=dtype, device=flag_gems.device)
    inp = base[:, :, ::2, :]
    assert not inp.is_contiguous()
    C = inp.shape[1]
    weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )
    eps = 1e-5

    ref_out = _ref_batch_norm(
        utils.to_reference(inp, True),
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        utils.to_reference(running_mean, True),
        utils.to_reference(running_var, True),
        eps,
    )

    # Calls the Ascend kernel directly so the strided input actually reaches
    # the offsets being fixed here, rather than the generic implementation.
    res_out, _, _, _ = ascend_bn(inp, weight, bias, running_mean, running_var, 0.1, eps)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="shape validation is specific to the Ascend kernel",
)
@pytest.mark.batch_norm_no_update
@pytest.mark.parametrize("bad_arg", ["weight", "bias", "running_mean", "running_var"])
def test_batch_norm_no_update_invalid_stats_shape(bad_arg):
    from flag_gems.runtime.backend._ascend.ops import batch_norm_no_update as ascend_bn

    # A per-channel tensor shorter than C used to be read out of bounds on the
    # device; F.batch_norm raises a deterministic shape error instead.
    dtype = torch.float32
    C = 3
    inp = torch.randn(size=(2, C, 4, 4), dtype=dtype, device=flag_gems.device)
    args = {
        name: torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
        for name in ("weight", "bias", "running_mean")
    }
    args["running_var"] = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )
    args[bad_arg] = args[bad_arg][: C - 1]

    with pytest.raises(RuntimeError):
        ascend_bn(
            inp,
            args["weight"],
            args["bias"],
            args["running_mean"],
            args["running_var"],
            0.1,
            1e-5,
        )
