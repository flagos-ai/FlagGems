import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8, 16])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason=(
        "PyTorch semantics issue (backend independent; reproduces with plain CPU tensors): "
        "aten::_embedding_bag with mode=0 (SUM) does not produce offset2bag by design (it returns "
        "an empty tensor with bag_size all 0), while aten::_embedding_bag_dense_backward in SUM "
        "mode needs offset2bag to map each sample back to its bag, so it necessarily raises "
        "IndexError: index out of range in self. Measured: (1) the same inputs raise the same "
        "error on torch CPU; (2) feeding a correct offset2bag by hand makes the reference "
        "computable and consistent with the flag_gems kernel; (3) the 12 analogous mode=1 (MEAN) "
        "cases all pass. The reference is unusable, this is not a kernel numeric error, and "
        "--ref cpu / fallback_cpu=all_ops cannot work around it. Upstream can restore the "
        "coverage by switching these cases to mean mode or building offset2bag itself."
    ),
)
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
    res_out = flag_gems._embedding_bag_dense_backward(
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


@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason=(
        "Same as test_embedding_bag_dense_backward: this variant also takes offset2bag from a "
        "mode=0 (SUM) forward, where aten returns an empty offset2bag, so the backward reference "
        "necessarily hits IndexError (it reproduces on CPU as well); with_weights itself is not involved"
    ),
)
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
    res_out = flag_gems._embedding_bag_dense_backward(
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


@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
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
    res_out = flag_gems._embedding_bag_dense_backward(
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
