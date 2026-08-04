import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# Same-numel shape pairs that are NOT broadcast-compatible. These exercise the
# reshape (rather than broadcast) path of resize_as.
if cfg.QUICK_MODE:
    RESIZE_AS_NON_BROADCAST_SHAPES = [
        ((2, 3), (3, 2)),
    ]
else:
    RESIZE_AS_NON_BROADCAST_SHAPES = [
        ((2, 3), (3, 2)),
        ((4, 4), (8, 2)),
        ((2, 6), (3, 4)),
        ((2, 2, 3), (3, 2, 2)),
        ((12,), (3, 4)),
        ((3, 4), (4, 3)),
        ((1024, 1024), (2048, 512)),
    ]


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


@pytest.mark.resize_as
def test_resize_as_mismatched_numel():
    inp = torch.randn(3, 4, device=flag_gems.device)
    template = torch.randn(5, 5, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            inp.resize_as(template)


@pytest.mark.resize_as
@pytest.mark.parametrize("src_shape, tgt_shape", RESIZE_AS_NON_BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_as_non_broadcastable(src_shape, tgt_shape, dtype):
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    template = torch.randn(tgt_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_template = utils.to_reference(template)

    ref_out = ref_inp.resize_as(ref_template)
    with flag_gems.use_gems():
        res_out = inp.resize_as(template)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.resize_as_
@pytest.mark.parametrize("src_shape, tgt_shape", RESIZE_AS_NON_BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_as__non_broadcastable(src_shape, tgt_shape, dtype):
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    template = torch.randn(tgt_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_template = utils.to_reference(template)

    ref_inp.resize_as_(ref_template)
    with flag_gems.use_gems():
        inp.resize_as_(template)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.resize_as
def test_resize_as_empty():
    # Exercises the early-return branch for empty tensors (numel == 0).
    inp = torch.randn(0, 3, dtype=torch.float32, device=flag_gems.device)
    template = torch.randn(3, 0, dtype=torch.float32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_template = utils.to_reference(template)

    ref_out = ref_inp.resize_as(ref_template)
    with flag_gems.use_gems():
        res_out = inp.resize_as(template)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.resize_as
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_as_non_contiguous(dtype):
    # Exercises reshape's copy path on a non-contiguous source (transpose).
    inp = torch.randn(
        4, 3, dtype=dtype, device=flag_gems.device
    ).T  # (3, 4) non-contiguous
    template = torch.randn(2, 6, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_template = utils.to_reference(template)

    ref_out = ref_inp.resize_as(ref_template)
    with flag_gems.use_gems():
        res_out = inp.resize_as(template)

    utils.gems_assert_equal(res_out, ref_out)
    assert not inp.is_contiguous(), "input must remain non-contiguous"


@pytest.mark.resize_as
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_as_non_contiguous_template(dtype):
    # Exercises the output allocation path when the template is non-contiguous:
    # the result must follow the template's shape (not its strides).
    inp = torch.randn(2, 6, dtype=dtype, device=flag_gems.device)
    template = torch.randn(
        3, 4, dtype=dtype, device=flag_gems.device
    ).T  # (4, 3) non-contiguous

    ref_inp = utils.to_reference(inp)
    ref_template = utils.to_reference(template)

    ref_out = ref_inp.resize_as(ref_template)
    with flag_gems.use_gems():
        res_out = inp.resize_as(template)

    utils.gems_assert_equal(res_out, ref_out)
    assert not template.is_contiguous(), "template must remain non-contiguous"
