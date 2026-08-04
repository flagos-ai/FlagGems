import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8, 16])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_dense_backward(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_dense_backward accuracy."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_weights = 50
    num_samples = num_bags * 3  # Average 3 samples per bag

    # Create inputs
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
        device=flag_gems.device,
    )

    # Forward pass to get offset2bag, bag_size, maximum_indices
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, False, None, False, -1
    )

    # Compute backward
    grad = torch.randn_like(output)

    ref_out = utils.to_reference(
        torch.ops.aten._embedding_bag_dense_backward(
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

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_dense_backward_with_weights(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_dense_backward accuracy with per_sample_weights."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_weights = 50
    num_samples = num_bags * 3

    # Create inputs
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
    per_sample_weights = torch.rand(num_samples, dtype=dtype, device=flag_gems.device)

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
        device=flag_gems.device,
    )

    # Forward pass
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, False, per_sample_weights, False, -1
    )

    # Compute backward
    grad = torch.randn_like(output)

    ref_out = utils.to_reference(
        torch.ops.aten._embedding_bag_dense_backward(
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

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag_dense_backward_mode_sum(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_dense_backward accuracy with mode=1 (sum)."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_weights = 50
    num_samples = num_bags * 3

    # Create inputs
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
        device=flag_gems.device,
    )

    # Forward pass with mode=1 (sum)
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 1, False, None, False, -1
    )

    # Compute backward
    grad = torch.randn_like(output)

    ref_out = utils.to_reference(
        torch.ops.aten._embedding_bag_dense_backward(
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

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8, 16])
@pytest.mark.parametrize("embedding_dim", [16, 32, 200])
# no bfloat16: eager embedding_bag_backward_cuda_max lacks a BFloat16 kernel,
# so there is no reference to compare against
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_embedding_bag_dense_backward_mode_max(num_bags, embedding_dim, dtype):
    """Test _embedding_bag_dense_backward accuracy with mode=2 (max).

    Max mode routes, per feature, the whole gradient of a bag to the row that
    won the max for that feature — a different law in kind from sum/mean, and
    the only one that reads maximum_indices. embedding_dim 200 exceeds the
    kernel's feature block, so the multi-block grid is exercised too.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_weights = 50
    num_samples = num_bags * 3

    # Create inputs
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
        device=flag_gems.device,
    )

    # Forward pass with mode=2 (max) supplies the argmax table
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 2, False, None, False, -1
    )

    # Compute backward
    grad = torch.randn_like(output)

    ref_out = utils.to_reference(
        torch.ops.aten._embedding_bag_dense_backward(
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            False,
            2,
            None,
            -1,
        )
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
            2,
            None,
            -1,
        )

    # Up to num_bags gradients can accumulate into one grad_weight cell (the
    # kernel accumulates them in fp32, eager in the storage dtype), so scale
    # the tolerance with that reduction depth.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=num_bags)
