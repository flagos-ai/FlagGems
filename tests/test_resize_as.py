import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.resize_as
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_as(shape, dtype):
    # resize_as requires same number of elements
    # Create target shape with same number of elements but different shape
    numel = 1
    for s in shape:
        numel *= s
    # Use various reshapes while keeping same numel
    target_shapes = [
        (numel,),
        (1, numel) if numel > 1 else (1,),
        (numel, 1) if numel > 1 else (1, 1),
    ]
    for target_shape in target_shapes:
        if target_shape == shape:
            continue
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        template = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)

        ref_inp = utils.to_reference(inp)
        ref_template = utils.to_reference(template)

        ref_out = ref_inp.resize_as(ref_template)
        with flag_gems.use_gems():
            res_out = inp.resize_as(template)

        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.resize_as_
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_as_(shape, dtype):
    numel = 1
    for s in shape:
        numel *= s
    target_shapes = [
        (numel,),
        (1, numel) if numel > 1 else (1,),
        (numel, 1) if numel > 1 else (1, 1),
    ]
    for target_shape in target_shapes:
        if target_shape == shape:
            continue
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        template = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)

        ref_inp = utils.to_reference(inp.clone())
        ref_template = utils.to_reference(template)

        ref_inp.resize_as_(ref_template)
        with flag_gems.use_gems():
            inp.resize_as_(template)

        utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.resize_as_mismatched_numel
def test_resize_as_mismatched_numel():
    inp = torch.randn(3, 4, device=flag_gems.device)
    template = torch.randn(5, 5, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            inp.resize_as(template)
