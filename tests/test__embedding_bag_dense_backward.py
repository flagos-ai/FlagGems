import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Embedding bag shapes for testing
EMBEDDING_BAG_SHAPES = [
    (3, 4),  # (num_bags, embedding_dim)
    (8, 16),
    (16, 32),
    (32, 64),
]

EMBEDDING_BAG_NUM_WEIGHTS_LIST = [10, 50, 100]
EMBEDDING_BAG_NUM_SAMPLES_LIST = [5, 20, 50]


def generate_embedding_bag_inputs(
    num_bags, embedding_dim, num_weights, num_samples, dtype, device
):
    """Generate inputs for _embedding_bag_dense_backward test."""
    # Create weight (embedding table)
    weight = torch.randn(num_weights, embedding_dim, dtype=dtype, device=device)

    # Create indices (num_samples indices into the embedding table)
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=device
    )

    # Create offsets (starting positions of each bag)
    # bags have varying sizes
    bag_sizes = torch.randint(1, 5, (num_bags,)).tolist()
    # Make sure the total matches num_samples
    while sum(bag_sizes) > num_samples:
        bag_sizes = [min(bs, sum(bag_sizes) - num_samples + 1) for bs in bag_sizes]
        if sum(bag_sizes) <= num_samples:
            break
        bag_sizes[-1] = (
            sum(bag_sizes[: len(bag_sizes) - 1]) - num_samples + bag_sizes[-1]
        )

    # Adjust to match exactly
    total = sum(bag_sizes)
    if total < num_samples:
        bag_sizes[-1] += num_samples - total

    offsets = [0]
    for bs in bag_sizes:
        offsets.append(offsets[-1] + bs)
    offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)

    # Create per_sample_weights (optional, sometimes None)
    per_sample_weights = torch.rand(num_samples, dtype=dtype, device=device)

    return weight, indices, offsets, per_sample_weights


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8, 16])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accuracy__embedding_bag_dense_backward(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_dense_backward accuracy."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = flag_gems.device
    num_weights = 50
    num_samples = num_bags * 3  # Average 3 samples per bag

    # Create inputs
    weight = torch.randn(
        num_weights, embedding_dim, dtype=dtype, device=device, requires_grad=True
    )
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=device
    )

    # Create offsets with varying bag sizes
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
        device=device,
    )

    # Forward pass to get offset2bag, bag_size, maximum_indices
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, False, None, False, -1
    )

    # Compute backward
    grad = torch.randn_like(output)

    ref_out = torch.ops.aten._embedding_bag_dense_backward(
        grad,
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        False,
        0,
        None,
        -1,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._embedding_bag_dense_backward(
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            False,
            0,
            None,
            -1,
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accuracy__embedding_bag_dense_backward_with_weights(
    num_bags, embedding_dim, dtype
):
    """Test _embedding_bag_dense_backward accuracy with per_sample_weights."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = flag_gems.device
    num_weights = 50
    num_samples = num_bags * 3

    # Create inputs
    weight = torch.randn(
        num_weights, embedding_dim, dtype=dtype, device=device, requires_grad=True
    )
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=device
    )
    per_sample_weights = torch.rand(num_samples, dtype=dtype, device=device)

    # Create offsets
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
        device=device,
    )

    # Forward pass
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, False, per_sample_weights, False, -1
    )

    # Compute backward
    grad = torch.randn_like(output)

    ref_out = torch.ops.aten._embedding_bag_dense_backward(
        grad,
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        False,
        0,
        per_sample_weights,
        -1,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._embedding_bag_dense_backward(
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            False,
            0,
            per_sample_weights,
            -1,
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accuracy__embedding_bag_dense_backward_mode_sum(
    num_bags, embedding_dim, dtype
):
    """Test _embedding_bag_dense_backward accuracy with mode=1 (sum)."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = flag_gems.device
    num_weights = 50
    num_samples = num_bags * 3

    # Create inputs
    weight = torch.randn(
        num_weights, embedding_dim, dtype=dtype, device=device, requires_grad=True
    )
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=device
    )

    # Create offsets
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
        device=device,
    )

    # Forward pass with mode=1 (sum)
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 1, False, None, False, -1
    )

    # Compute backward
    grad = torch.randn_like(output)

    ref_out = torch.ops.aten._embedding_bag_dense_backward(
        grad,
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        False,
        1,
        None,
        -1,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._embedding_bag_dense_backward(
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            False,
            1,
            None,
            -1,
        )

    gems_assert_close(res_out, ref_out, dtype)
