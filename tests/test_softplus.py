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


@pytest.mark.softplus
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    beta = torch.rand(1).item()
    threshold = torch.rand(1).item() * 40.0
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.nn.functional.softplus(ref_inp, beta=beta, threshold=threshold)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.softplus(inp, beta=beta, threshold=threshold)

    utils.gems_assert_close(res_out, ref_out, dtype)


def _softplus_backward_raw_cpu(value):
    return torch.empty(0, dtype=value.dtype, device="cpu").set_(
        value.untyped_storage().cpu(),
        value.storage_offset(),
        value.shape,
        value.stride(),
    )


def _softplus_backward_snapshot(value):
    return (
        value.device,
        value.dtype,
        tuple(value.shape),
        value.stride(),
        value.storage_offset(),
        value.untyped_storage()._cdata,
        value.requires_grad,
        value.untyped_storage().data_ptr(),
        value.untyped_storage().nbytes(),
        value.is_conj(),
        value.is_neg(),
    ), _softplus_backward_raw_cpu(value).contiguous().reshape(-1).view(
        torch.uint8
    ).clone()


def _checked_softplus_backward(grad, inp, beta, threshold):
    snapshots = [_softplus_backward_snapshot(value) for value in (grad, inp)]
    try:
        result = flag_gems.softplus_backward(grad, inp, beta=beta, threshold=threshold)
    finally:
        for value, (metadata, contents) in zip((grad, inp), snapshots):
            after_metadata, after_contents = _softplus_backward_snapshot(value)
            assert after_metadata == metadata, "candidate changed input metadata"
            assert torch.equal(
                after_contents, contents
            ), "candidate changed input values"
    assert isinstance(result, torch.Tensor)
    assert result.device == inp.device and result.dtype == inp.dtype
    assert result.shape == inp.shape
    assert all(
        result.untyped_storage()._cdata != metadata[5] for metadata, _ in snapshots
    )
    storage = result.untyped_storage()
    if result.numel() and storage.nbytes():
        output_start = storage.data_ptr()
        output_end = output_start + storage.nbytes()
        for metadata, _ in snapshots:
            if result.device == metadata[0] and metadata[8]:
                input_start = metadata[7]
                input_end = input_start + metadata[8]
                assert (
                    output_end <= input_start or input_end <= output_start
                ), "candidate output storage must not overlap either input"
    return result


@pytest.mark.softplus_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad_output = torch.randn_like(res_inp)
    beta = torch.rand(1).item() + 0.5  # Ensure beta > 0.5 for stability
    threshold = torch.rand(1).item() * 40.0

    # Follow the native Float32 opmath at the piecewise threshold; promoting
    # the reference to Double can select a different branch.
    ref_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    ref_inp = utils.to_reference(res_inp, upcast=False).to(dtype=ref_dtype)
    ref_grad_output = utils.to_reference(res_grad_output, upcast=False).to(
        dtype=ref_dtype
    )

    ref_grad_input = torch.ops.aten.softplus_backward(
        ref_grad_output, ref_inp, beta=beta, threshold=threshold
    )
    res_grad_input = _checked_softplus_backward(
        res_grad_output, res_inp, beta=beta, threshold=threshold
    )

    utils.gems_assert_close(
        res_grad_input.detach().cpu(), ref_grad_input.detach().cpu(), dtype
    )
