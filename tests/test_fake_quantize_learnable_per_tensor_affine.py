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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Common quantization parameter sets: (quant_min, quant_max)
QUANT_RANGES = [
    (0, 255),  # uint8 affine
    (-128, 127),  # int8 symmetric / affine
]


def _make_inputs(shape, dtype, quant_min, quant_max):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 5
    # scale is a single-element float tensor; zero_point is a single-element
    # integer tensor, matching the aten signature.
    scale = torch.tensor([0.1], dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.tensor(
        [(quant_min + quant_max) // 2], dtype=torch.int32, device=flag_gems.device
    )
    return inp, scale, zero_point


def _snapshot(tensor):
    storage = tensor.untyped_storage()
    metadata = (
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
        storage._cdata,
        storage.data_ptr(),
        storage.nbytes(),
    )
    raw = (
        torch.empty(0, dtype=torch.uint8)
        .set_(storage.cpu(), 0, (storage.nbytes(),), (1,))
        .clone()
    )
    return metadata, raw


_OMITTED_GRAD_FACTOR = object()


def _run_with_contract(
    inp, scale, zero_point, quant_min, quant_max, grad_factor=_OMITTED_GRAD_FACTOR
):
    inputs = [inp, scale, zero_point]
    snapshots = [_snapshot(value) for value in inputs]
    if grad_factor is _OMITTED_GRAD_FACTOR:
        result = flag_gems._fake_quantize_learnable_per_tensor_affine(
            inp, scale, zero_point, quant_min, quant_max
        )
    else:
        result = flag_gems._fake_quantize_learnable_per_tensor_affine(
            inp, scale, zero_point, quant_min, quant_max, grad_factor
        )
    assert isinstance(result, torch.Tensor)
    assert result.device == inp.device
    assert result.dtype == inp.dtype
    assert result.shape == inp.shape
    for value, (metadata, raw) in zip(inputs, snapshots):
        current_metadata, current_raw = _snapshot(value)
        assert current_metadata == metadata
        assert torch.equal(current_raw, raw)
        if result.numel() and metadata[10]:
            storage = result.untyped_storage()
            assert storage._cdata != metadata[8]
            start, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
            original_start, original_end = metadata[9], metadata[9] + metadata[10]
            assert end <= original_start or original_end <= start
    return result


@pytest.mark.fake_quantize_learnable_per_tensor_affine
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("quant_range", QUANT_RANGES)
def test_fake_quantize_learnable_per_tensor_affine(shape, dtype, quant_range):
    quant_min, quant_max = quant_range
    inp, scale, zero_point = _make_inputs(shape, dtype, quant_min, quant_max)
    ref_inp = utils.to_reference(inp)
    ref_scale = utils.to_reference(scale)
    ref_zp = utils.to_reference(zero_point)

    ref_out = torch.ops.aten._fake_quantize_learnable_per_tensor_affine(
        ref_inp, ref_scale, ref_zp, quant_min, quant_max
    )
    res_out = _run_with_contract(inp, scale, zero_point, quant_min, quant_max)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fake_quantize_learnable_per_tensor_affine
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fake_quantize_learnable_per_tensor_affine_grad_factor(shape, dtype):
    # grad_factor does not affect the forward output; verify it is accepted.
    quant_min, quant_max = 0, 255
    inp, scale, zero_point = _make_inputs(shape, dtype, quant_min, quant_max)
    ref_inp = utils.to_reference(inp)
    ref_scale = utils.to_reference(scale)
    ref_zp = utils.to_reference(zero_point)

    ref_out = torch.ops.aten._fake_quantize_learnable_per_tensor_affine(
        ref_inp, ref_scale, ref_zp, quant_min, quant_max, 0.5
    )
    res_out = _run_with_contract(inp, scale, zero_point, quant_min, quant_max, 0.5)

    utils.gems_assert_close(res_out, ref_out, dtype)
