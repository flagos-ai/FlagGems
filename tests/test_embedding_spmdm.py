# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Cross-platform accuracy tests for :func:`flag_gems.embedding_spmdm`."""

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SHAPES = (
    (7, 1, 1, 1),
    (17, 24, 3, 5),
    (257, 64, 8, 16),
    (1024, 128, 4, 33),
)


def _make_inputs(shape, dtype, index_dtype, weighted, noncontiguous=False):
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
            dtype=index_dtype,
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
            dtype=index_dtype,
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


def _reference(table, indices, weights):
    ref_table = utils.to_reference(table, False)
    ref_indices = utils.to_reference(indices, False)
    ref_weights = utils.to_reference(weights, False)
    return F.embedding_bag(
        ref_indices,
        ref_table,
        offsets=None,
        mode="sum",
        per_sample_weights=ref_weights,
        max_norm=None,
        scale_grad_by_freq=False,
        sparse=False,
        include_last_offset=False,
        padding_idx=None,
    )


def _assert_close(actual, expected, dtype):
    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.embedding_spmdm
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_embedding_spmdm_matches_embedding_bag(shape, dtype, weighted):
    table, indices, weights = _make_inputs(shape, dtype, torch.int64, weighted)
    actual = flag_gems.embedding_spmdm(table, indices, weights)
    expected = _reference(table, indices, weights)
    assert actual.shape == (shape[2], shape[1])
    assert actual.dtype == dtype
    _assert_close(actual, expected, dtype)


@pytest.mark.embedding_spmdm
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_embedding_spmdm_index_dtypes(index_dtype):
    shape = (31, 32, 4, 7)
    table, indices, weights = _make_inputs(shape, torch.float32, index_dtype, True)
    actual = flag_gems.embedding_spmdm(table, indices, weights)
    expected = _reference(table, indices, weights)
    _assert_close(actual, expected, torch.float32)


@pytest.mark.embedding_spmdm
def test_embedding_spmdm_noncontiguous_inputs():
    shape = (37, 40, 3, 9)
    table, indices, weights = _make_inputs(
        shape, torch.float16, torch.int64, True, noncontiguous=True
    )
    assert not table.is_contiguous()
    assert not indices.is_contiguous()
    assert not weights.is_contiguous()
    actual = flag_gems.embedding_spmdm(table, indices, weights)
    expected = _reference(table, indices, weights)
    _assert_close(actual, expected, torch.float16)


@pytest.mark.embedding_spmdm
def test_embedding_spmdm_validation():
    table = torch.randn(11, 16, device=flag_gems.device)
    indices = torch.zeros(2, 3, dtype=torch.int64, device=flag_gems.device)
    weights = torch.ones(2, 3, device=flag_gems.device)

    with pytest.raises(ValueError, match="2-dimensional"):
        flag_gems.embedding_spmdm(table.unsqueeze(0), indices)
    with pytest.raises(TypeError, match="int32 or torch.int64"):
        flag_gems.embedding_spmdm(table, indices.float())
    with pytest.raises(ValueError, match="same shape"):
        flag_gems.embedding_spmdm(table, indices, weights[:, :-1])
    with pytest.raises(TypeError, match="same dtype"):
        flag_gems.embedding_spmdm(table.half(), indices, weights)
