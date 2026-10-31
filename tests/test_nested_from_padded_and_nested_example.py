import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_inputs(B, T, trailing_dims, ragged_idx, dtype, device, seed=42):
    """Create a padded tensor and the nested example describing the layout.

    ``padded`` has shape ``[B, T, *trailing_dims]`` (with singleton dims
    inserted so the ragged dim sits at ``ragged_idx``); the example is built
    from variable-length slices of it, exactly how the op is used in practice.
    """
    shape = [B] + [1] * (ragged_idx - 1) + [T] + list(trailing_dims)
    padded = torch.randn(shape, dtype=dtype, device=device)

    np.random.seed(seed)
    lengths = np.random.randint(1, T + 1, size=B)
    components = []
    for b, length in enumerate(lengths):
        # Slice the ragged dim of padded[b] (shifted by the inserted dims).
        index = [slice(None)] * padded.dim()
        index[ragged_idx] = slice(0, int(length))
        components.append(padded[tuple([b] + index[1:])])
    example = torch.nested.nested_tensor(components, device=device)
    return padded, example


def _assert_nested_close(res, ref, dtype):
    assert res.is_nested
    assert ref.is_nested

    res_unbind = torch.unbind(res)
    ref_unbind = torch.unbind(ref)

    assert len(res_unbind) == len(ref_unbind)
    for res_t, ref_t in zip(res_unbind, ref_unbind):
        assert res_t.shape == ref_t.shape
        utils.gems_assert_close(res_t, ref_t, dtype)


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("B,T,D", [(2, 8, 4), (8, 32, 16), (3, 5, 4)])
def test_nested_from_padded_and_nested_example_3d(dtype, B, T, D):
    # The nested result's structure is device-independent but the composed
    # values follow the input device; the CPU reference stays value-comparable,
    # so no skip is needed (verified against the --ref=cpu pass).
    padded, example = _make_inputs(B, T, (D,), 1, dtype, flag_gems.device)

    ref_padded = utils.to_reference(padded)
    ref_example = utils.to_reference(example)
    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        ref_padded, ref_example
    )

    res_out = flag_gems._nested_from_padded_and_nested_example(padded, example)

    _assert_nested_close(res_out, ref_out, dtype)
    # The output structure is taken verbatim from the example's metadata.
    assert torch.equal(res_out._nested_tensor_size(), ref_out._nested_tensor_size())
    assert torch.equal(
        res_out._nested_tensor_strides(), ref_out._nested_tensor_strides()
    )
    assert torch.equal(
        res_out._nested_tensor_storage_offsets(),
        ref_out._nested_tensor_storage_offsets(),
    )


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("B,T", [(2, 8), (8, 32), (3, 5)])
def test_nested_from_padded_and_nested_example_2d(dtype, B, T):
    padded, example = _make_inputs(B, T, (), 1, dtype, flag_gems.device)

    ref_padded = utils.to_reference(padded)
    ref_example = utils.to_reference(example)
    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        ref_padded, ref_example
    )

    res_out = flag_gems._nested_from_padded_and_nested_example(padded, example)

    _assert_nested_close(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_and_nested_example_ragged_idx(dtype):
    # ragged_idx=2 on a 5D padded tensor [B, H, T, D]: components are (H, len, D).
    B, T, D = 2, 4, 2
    padded, example = _make_inputs(B, T, (D,), 2, dtype, flag_gems.device)
    # _make_inputs only inserts singletons before the ragged dim; rebuild with
    # an explicit leading H dim inside each component.
    np.random.seed(42)
    lengths = np.random.randint(1, T + 1, size=B)
    components = [padded[b, :, : int(l)] for b, l in enumerate(lengths)]
    example = torch.nested.nested_tensor(components, device=flag_gems.device)

    ref_padded = utils.to_reference(padded)
    ref_example = utils.to_reference(example)
    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        ref_padded, ref_example
    )

    res_out = flag_gems._nested_from_padded_and_nested_example(padded, example)

    _assert_nested_close(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_and_nested_example_zero_length(dtype):
    # A zero-length component contributes no elements to the packed buffer.
    B, T, D = 3, 4, 2
    padded = torch.randn(B, T, D, dtype=dtype, device=flag_gems.device)
    example = torch.nested.nested_tensor(
        [padded[0, :3], padded[1, :0], padded[2, :4]], device=flag_gems.device
    )

    ref_padded = utils.to_reference(padded)
    ref_example = utils.to_reference(example)
    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        ref_padded, ref_example
    )

    res_out = flag_gems._nested_from_padded_and_nested_example(padded, example)

    _assert_nested_close(res_out, ref_out, dtype)
