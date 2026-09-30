import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _assert_unique2(res, ref, dtype):
    """Compare all three aten::_unique2 outputs directly.

    aten::_unique2 always returns three tensors ``(output, inverse_indices,
    counts)``; ``inverse_indices``/``counts`` are empty tensors (not ``None``)
    when not requested. flag_gems mirrors that contract, so every output is
    compared with the reference unconditionally.
    """
    res_out, res_inverse, res_counts = res
    ref_out, ref_inverse, ref_counts = ref

    if dtype in utils.ALL_FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(res_out, ref_out)

    utils.gems_assert_equal(res_inverse, ref_inverse)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize(
    "return_inverse, return_counts",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_unique2(shape, dtype, return_inverse, return_counts):
    """Basic _unique2 coverage across all return-flag combinations."""
    res_inp = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref = torch._unique2(
        ref_inp, return_inverse=return_inverse, return_counts=return_counts
    )
    res = flag_gems._unique2(
        res_inp, return_inverse=return_inverse, return_counts=return_counts
    )
    _assert_unique2(res, ref, dtype)


@pytest.mark.unique2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize(
    "return_inverse, return_counts",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_unique2_empty_input(dtype, return_inverse, return_counts):
    """Zero-size parity: empty input must yield empty outputs for every combo.

    The small-input kernel path has no zero-size guard, so this boundary needs
    explicit coverage across all return_inverse/return_counts combinations.
    """
    res_inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref = torch._unique2(
        ref_inp, return_inverse=return_inverse, return_counts=return_counts
    )
    res = flag_gems._unique2(
        res_inp, return_inverse=return_inverse, return_counts=return_counts
    )

    res_out, res_inverse, res_counts = res
    assert res_out.numel() == 0
    assert res_inverse.numel() == 0
    assert res_counts.numel() == 0
    _assert_unique2(res, ref, dtype)


@pytest.mark.unique2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize(
    "return_inverse, return_counts",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_unique2_large_reduce(dtype, return_inverse, return_counts):
    """Large-input coverage past the numel() <= 8192 small-path cutoff.

    8193 elements force the large-input branches: the return_inverse path uses
    sorted_indices_unique_flat while the others use sorted_quick_unique_flat, so
    both large branches are exercised with and without inverse/counts.
    """
    shape = (8193,)
    res_inp = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref = torch._unique2(
        ref_inp, return_inverse=return_inverse, return_counts=return_counts
    )
    res = flag_gems._unique2(
        res_inp, return_inverse=return_inverse, return_counts=return_counts
    )
    _assert_unique2(res, ref, dtype)


@pytest.mark.unique2
def test_unique2_edge_cases():
    """All-same and all-unique inputs."""
    for factory in (
        lambda: torch.ones(100, dtype=torch.int64, device=flag_gems.device),
        lambda: torch.arange(100, dtype=torch.int64, device=flag_gems.device),
    ):
        res_inp = factory()
        ref_inp = utils.to_reference(res_inp)

        ref = torch._unique2(ref_inp, return_inverse=True, return_counts=True)
        res = flag_gems._unique2(res_inp, return_inverse=True, return_counts=True)
        _assert_unique2(res, ref, torch.int64)


@pytest.mark.unique2
def test_unique2_float_special_values():
    """Repeated NaN, signed zero and non-integral floating values.

    Includes capability-gated FP64 coverage where the device supports it.
    Each NaN compares unequal to every other value, so aten::_unique2 keeps one
    unique slot per NaN occurrence; which slot a given NaN maps to in
    ``inverse_indices`` is therefore implementation-defined. We validate the
    unique values and counts directly and check inverse_indices by
    reconstructing the input (out[inverse] == input, NaN-aware).
    """
    dtypes = [torch.float32]
    if utils.fp64_is_supported:
        dtypes.append(torch.float64)

    values = [
        1.5,
        -2.25,
        0.0,
        -0.0,
        float("inf"),
        -2.25,
        float("nan"),
        1.5,
        float("nan"),
        float("-inf"),
        float("nan"),
        3.75,
    ]

    for dtype in dtypes:
        res_inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(res_inp)

        ref_out, ref_inverse, ref_counts = torch._unique2(
            ref_inp, return_inverse=True, return_counts=True
        )
        res_out, res_inverse, res_counts = flag_gems._unique2(
            res_inp, return_inverse=True, return_counts=True
        )

        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
        utils.gems_assert_equal(res_counts, ref_counts)

        # inverse_indices must scatter the unique values back to the input.
        recon = res_out[res_inverse]
        utils.gems_assert_close(recon, ref_inp, dtype, equal_nan=True)
