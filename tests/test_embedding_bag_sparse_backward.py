import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_inputs(num_bags, embedding_dim, num_weights, dtype, with_psw=False):
    num_samples = num_bags * 3

    weight = torch.randn(
        num_weights,
        embedding_dim,
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=flag_gems.device
    )

    # Build offsets with varying (non-empty) bag sizes distributions.
    bag_sizes_list = []
    remaining = num_samples
    for _ in range(num_bags - 1):
        bs = max(1, remaining // 2)
        bag_sizes_list.append(min(bs, remaining - num_bags + 1))
        remaining -= bag_sizes_list[-1]
    bag_sizes_list.append(remaining)
    offsets = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(bag_sizes_list), dim=0).tolist())[:-1],
        dtype=torch.long,
        device=flag_gems.device,
    )

    per_sample_weights = None
    if with_psw:
        per_sample_weights = torch.rand(
            num_samples, dtype=dtype, device=flag_gems.device
        )

    return weight, indices, offsets, per_sample_weights, num_samples


def _run_sparse_backward(
    grad,
    indices,
    offsets,
    offset2bag,
    bag_size,
    num_weights,
    mode,
    per_sample_weights,
    padding_idx,
):
    ref_out = utils.to_reference(
        torch.ops.aten._embedding_bag_sparse_backward(
            grad,
            indices,
            offsets,
            offset2bag,
            bag_size,
            num_weights,
            False,
            mode,
            per_sample_weights,
            padding_idx,
        )
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._embedding_bag_sparse_backward(
            grad,
            indices,
            offsets,
            offset2bag,
            bag_size,
            num_weights,
            False,
            mode,
            per_sample_weights,
            padding_idx,
        )
    utils.gems_assert_close(res_out, ref_out, dtype=grad.dtype)


# Hardcoded bag counts covering small/medium batch sizes for sparse backward coverage.
@pytest.mark.embedding_bag_sparse_backward
@pytest.mark.parametrize("num_bags", [3, 8, 16])
# Hardcoded embedding dims covering typical small/medium embedding vector widths.
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_sparse_backward_sum(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_sparse_backward accuracy in sum mode (mode=0)."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # num_weights hardcoded to a small value (50) to keep test runtime bounded.
    num_weights = 50
    weight, indices, offsets, _, _ = _make_inputs(
        num_bags, embedding_dim, num_weights, dtype
    )

    # Forward pass with sparse=True to obtain offset2bag / bag_size.
    output, offset2bag, bag_size, _ = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, True, None, False, -1
    )

    grad = torch.randn_like(output)
    _run_sparse_backward(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        num_weights,
        mode=0,
        per_sample_weights=None,
        padding_idx=-1,
    )


@pytest.mark.embedding_bag_sparse_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_sparse_backward_mean(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_sparse_backward accuracy in mean mode (mode=1)."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # num_weights hardcoded to a small value (50) to keep test runtime bounded.
    num_weights = 50
    weight, indices, offsets, _, _ = _make_inputs(
        num_bags, embedding_dim, num_weights, dtype
    )

    output, offset2bag, bag_size, _ = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 1, True, None, False, -1
    )

    grad = torch.randn_like(output)
    _run_sparse_backward(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        num_weights,
        mode=1,
        per_sample_weights=None,
        padding_idx=-1,
    )


@pytest.mark.embedding_bag_sparse_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_sparse_backward_per_sample_weights(
    num_bags, embedding_dim, dtype
):
    """Test _embedding_bag_sparse_backward accuracy with per_sample_weights (sum mode)."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # num_weights hardcoded to a small value (50) to keep test runtime bounded.
    num_weights = 50
    weight, indices, offsets, per_sample_weights, _ = _make_inputs(
        num_bags, embedding_dim, num_weights, dtype, with_psw=True
    )

    output, offset2bag, bag_size, _ = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, True, per_sample_weights, False, -1
    )

    grad = torch.randn_like(output)
    _run_sparse_backward(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        num_weights,
        mode=0,
        per_sample_weights=per_sample_weights,
        padding_idx=-1,
    )


@pytest.mark.embedding_bag_sparse_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_sparse_backward_padding_idx(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_sparse_backward accuracy with padding_idx filtering."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # num_weights hardcoded to a small value (50) to keep test runtime bounded.
    num_weights = 50
    weight, indices, offsets, _, _ = _make_inputs(
        num_bags, embedding_dim, num_weights, dtype
    )

    # Force one of the indices to be the padding index so the filter path is exercised.
    padding_idx = int(indices[0].item())
    output, offset2bag, bag_size, _ = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, True, None, False, padding_idx
    )

    grad = torch.randn_like(output)
    _run_sparse_backward(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        num_weights,
        mode=0,
        per_sample_weights=None,
        padding_idx=padding_idx,
    )
