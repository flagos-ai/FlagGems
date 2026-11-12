# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Cross-platform accuracy tests for :func:`flag_gems.embedding_spmdm_grad`."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SHAPES = (
    (7, 1, 1, 1),
    (17, 24, 3, 5),
    (257, 64, 8, 16),
    (1024, 128, 4, 33),
)


def _make_inputs(shape, dtype, weighted, noncontiguous=False):
    num_embeddings, embedding_dim, batch_size, bag_size = shape
    torch.manual_seed(2026)
    if noncontiguous:
        table = torch.randn(
            embedding_dim,
            num_embeddings,
            dtype=dtype,
            device=flag_gems.device,
        ).T
        indices = torch.randint(
            0,
            num_embeddings,
            (batch_size, bag_size * 2),
            dtype=torch.int64,
            device="cpu",
        ).to(flag_gems.device)[:, ::2]
    else:
        table = torch.randn(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device=flag_gems.device,
        )
        indices = torch.randint(
            0,
            num_embeddings,
            (batch_size, bag_size),
            dtype=torch.int64,
            device="cpu",
        ).to(flag_gems.device)
    if bag_size > 1:
        indices[:, 1::3] = indices[:, :1].expand(-1, indices[:, 1::3].shape[1])
    weights = None
    if weighted:
        weight_bag_size = bag_size * 2 if noncontiguous else bag_size
        weights = torch.randn(
            batch_size,
            weight_bag_size,
            dtype=dtype,
            device=flag_gems.device,
        )
        if noncontiguous:
            weights = weights[:, ::2]
    return table, indices, weights


def _reference_grad_table(grad_output, indices, weights, num_embeddings):
    batch_size, bag_size = indices.shape
    embedding_dim = grad_output.shape[1]
    # Widen both operands to float32 *before* multiplying, matching the order
    # the kernel uses (it promotes every fp16/bf16 load to fp32 and only then
    # multiplies).  Multiplying in the input dtype first rounds the product back
    # to fp16/bf16 and injects a reference-side error of ~1 ULP per element,
    # which can exceed the tolerance and blame the kernel for the reference's
    # own rounding.
    sample_grad = (
        grad_output[:, None, :]
        .expand(batch_size, bag_size, embedding_dim)
        .to(torch.float32)
    )
    if weights is not None:
        sample_grad = sample_grad * weights[:, :, None].to(torch.float32)
    # Compute in float32 regardless of the input dtype.  CPU bf16/fp16
    # reductions have different rounding behavior than the GPU implementation
    # under test, so the reference needs to be strictly more precise than
    # the GPU path for the comparison to be meaningful.
    sample_grad_fp32 = sample_grad.reshape(-1, embedding_dim)
    flat_indices = indices.reshape(-1).to(torch.int64)
    return torch.ops.aten.embedding_dense_backward(
        sample_grad_fp32,
        flat_indices,
        num_embeddings,
        -1,
        False,
    ).to(grad_output.dtype)


def _reference_grad_weights(grad_output, table, indices):
    batch_size, bag_size = indices.shape
    flat_indices = indices.reshape(-1).to(torch.int64)
    expanded_grad = (
        grad_output[:, None, :]
        .expand(batch_size, bag_size, grad_output.shape[1])
        .reshape(-1, grad_output.shape[1])
    )
    selected_table = torch.index_select(table, 0, flat_indices)
    # Compute in float32 for the same reason as above.
    return (
        (selected_table.to(torch.float32) * expanded_grad.to(torch.float32))
        .sum(dim=1)
        .to(grad_output.dtype)
        .reshape_as(indices)
    )


def _reference_grad(grad_output, table, indices, weights):
    ref_grad_output = utils.to_reference(grad_output, False)
    ref_table = utils.to_reference(table, False)
    ref_indices = utils.to_reference(indices, False)
    ref_weights = utils.to_reference(weights, False)
    grad_table = _reference_grad_table(
        ref_grad_output, ref_indices, ref_weights, table.shape[0]
    )
    grad_weights = (
        _reference_grad_weights(ref_grad_output, ref_table, ref_indices)
        if weights is not None
        else None
    )
    return grad_table, grad_weights


def _assert_close(actual, expected, dtype, reduce_dim):
    atol = 1e-3 if dtype is torch.bfloat16 else 1e-4
    utils.gems_assert_close(actual, expected, dtype, atol=atol, reduce_dim=reduce_dim)


@pytest.mark.embedding_spmdm_grad
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_embedding_spmdm_grad_matches_reference(shape, dtype, weighted):
    table, indices, weights = _make_inputs(shape, dtype, weighted)
    grad_output = torch.randn(shape[2], shape[1], dtype=dtype, device=flag_gems.device)
    actual_table, actual_weights = flag_gems.embedding_spmdm_grad(
        grad_output, table, indices, weights
    )
    expected_table, expected_weights = _reference_grad(
        grad_output, table, indices, weights
    )
    assert actual_table.shape == table.shape
    assert actual_table.dtype == dtype
    # ``grad_table`` is computed by atomic-add accumulation across the
    # bag, so widen tolerance proportionally to the bag size.
    _assert_close(actual_table, expected_table, dtype, reduce_dim=shape[3])
    if weights is None:
        assert actual_weights is None
    else:
        assert actual_weights.shape == weights.shape
        assert actual_weights.dtype == dtype
        # ``grad_weights`` is a dot product along the embedding dim.
        _assert_close(actual_weights, expected_weights, dtype, reduce_dim=shape[1])


@pytest.mark.embedding_spmdm_grad
def test_embedding_spmdm_grad_noncontiguous_inputs():
    shape = (37, 40, 3, 9)
    table, indices, weights = _make_inputs(
        shape, torch.float16, True, noncontiguous=True
    )
    grad_output = torch.randn(
        shape[2], shape[1], dtype=torch.float16, device=flag_gems.device
    )
    actual = flag_gems.embedding_spmdm_grad(grad_output, table, indices, weights)
    expected = _reference_grad(grad_output, table, indices, weights)
    _assert_close(actual[0], expected[0], torch.float16, reduce_dim=shape[3])
    _assert_close(actual[1], expected[1], torch.float16, reduce_dim=shape[1])


@pytest.mark.embedding_spmdm_grad
def test_embedding_spmdm_grad_validation():
    table = torch.randn(11, 16, device=flag_gems.device)
    indices = torch.zeros(2, 3, dtype=torch.int64, device=flag_gems.device)
    weights = torch.ones(2, 3, device=flag_gems.device)
    with pytest.raises(ValueError, match="grad_output must have shape"):
        flag_gems.embedding_spmdm_grad(
            torch.randn(2, 15, device=flag_gems.device),
            table,
            indices,
            weights,
        )
