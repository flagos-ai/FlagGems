# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        combine_topk_swa_indices as vllm_combine_topk_swa_indices,
    )

    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = True
except Exception:
    vllm_combine_topk_swa_indices = None
    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = False


def _reference_combine_with_pair_metadata(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
    *,
    assume_ordered_topk=False,
):
    """CPU reference for both combined rows and encoded pair metadata."""
    topk_indices = topk_indices.detach().cpu()
    query_start_loc = query_start_loc.detach().cpu()
    seq_lens = seq_lens.detach().cpu()
    gather_lens = gather_lens.detach().cpu()

    num_tokens = topk_indices.shape[0]
    combined_topk = (topk + window_size + 127) // 128 * 128
    combined = torch.full((num_tokens, combined_topk), -1, dtype=torch.int32)
    lens = torch.empty(num_tokens, dtype=torch.int32)
    pair_metadata = torch.zeros((num_tokens + 1) // 2, dtype=torch.int32)

    base = int(query_start_loc[0].item())
    for batch_idx in range(seq_lens.numel()):
        query_start = int(query_start_loc[batch_idx].item()) - base
        query_end = int(query_start_loc[batch_idx + 1].item()) - base
        query_len = query_end - query_start
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = int(gather_lens[batch_idx].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len

        for token_idx in range(query_start, query_end):
            pos = start_pos + token_idx - query_start
            raw_topk_len = (pos + 1) // compress_ratio
            topk_len = min(raw_topk_len, topk)
            swa_len = min(pos + 1, window_size)
            valid_len = topk_len + swa_len

            combined[token_idx, :topk_len] = (
                topk_indices[token_idx, :topk_len] + M * batch_idx
            )
            if swa_len:
                swa = torch.arange(swa_len, dtype=torch.int32)
                combined[token_idx, topk_len:valid_len] = (
                    M * batch_idx + N + swa + pos - swa_len + 1 - gather_start
                )
            lens[token_idx] = valid_len

            if token_idx % 2 != 0 or token_idx + 1 >= query_end:
                continue

            next_pos = pos + 1
            next_raw_topk_len = (next_pos + 1) // compress_ratio
            next_topk_len = min(next_raw_topk_len, topk)
            valid_length_relation = next_topk_len in (
                topk_len,
                topk_len + 1,
            )
            if assume_ordered_topk:
                prefix_matches = (
                    next_raw_topk_len in (raw_topk_len, raw_topk_len + 1)
                    and next_raw_topk_len <= topk
                )
            else:
                prefix_matches = valid_length_relation and torch.equal(
                    topk_indices[token_idx, :topk_len],
                    topk_indices[token_idx + 1, :topk_len],
                )

            topk_grows = next_topk_len == topk_len + 1
            mode = 0
            if prefix_matches and valid_len > 0 and window_size > 0:
                if swa_len < window_size:
                    mode = 3 if topk_grows else 1
                elif min(next_pos + 1, window_size) == window_size:
                    mode = 4 if topk_grows else 2
            if mode:
                pair_metadata[token_idx // 2] = (topk_len << 3) | mode

    return combined, lens, pair_metadata


@pytest.mark.parametrize(
    (
        "topk_values",
        "query_start_values",
        "seq_len_values",
        "gather_len_values",
        "window_size",
        "compress_ratio",
        "topk",
        "M",
        "N",
    ),
    [
        (
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [0, 2, 3],
            [8, 10],
            [8, 10],
            4,
            2,
            4,
            64,
            16,
        ),
        (
            [
                [100, 101, 102, 103],
                [110, 111, 112, 113],
                [120, 121, 122, 123],
                [130, 131, 132, 133],
                [140, 141, 142, 143],
            ],
            [0, 3, 5],
            [6, 4],
            [4, 3],
            3,
            2,
            4,
            20,
            8,
        ),
    ],
)
@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_accuracy(
    topk_values,
    query_start_values,
    seq_len_values,
    gather_len_values,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    device = "cuda"
    topk_indices = torch.tensor(topk_values, device=device, dtype=torch.int32)
    query_start_loc = torch.tensor(query_start_values, device=device, dtype=torch.int32)
    seq_lens = torch.tensor(seq_len_values, device=device, dtype=torch.int32)
    gather_lens = torch.tensor(gather_len_values, device=device, dtype=torch.int32)

    actual, actual_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )

    expected = torch.full_like(actual, -1)
    expected_lens = torch.empty_like(actual_lens)
    for batch in range(seq_lens.numel()):
        start = int(query_start_loc[batch].item()) - int(query_start_loc[0].item())
        end = int(query_start_loc[batch + 1].item()) - int(query_start_loc[0].item())
        query_len = end - start
        seq_len = int(seq_lens[batch].item())
        gather_len = int(gather_lens[batch].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len
        for token_idx in range(start, end):
            token_in_query = token_idx - start
            pos = start_pos + token_in_query
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            if topk_len:
                expected[token_idx, :topk_len] = (
                    topk_indices[token_idx, :topk_len] + M * batch
                )
            for j in range(swa_len):
                expected[token_idx, topk_len + j] = (
                    M * batch + N + j + pos - swa_len + 1 - gather_start
                )
            expected_lens[token_idx] = topk_len + swa_len

    fg_testing.assert_equal(actual, expected)
    fg_testing.assert_equal(actual_lens, expected_lens)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not _HAS_VLLM_COMBINE_TOPK_SWA_INDICES),
    reason="requires cuda and vllm deepseek_v4_ops.combine_topk_swa_indices",
)
def test_combine_topk_swa_indices_vllm_accuracy():
    device = "cuda"
    topk_indices = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        device=device,
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([8, 10], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([8, 10], device=device, dtype=torch.int32)
    args = (topk_indices, query_start_loc, seq_lens, gather_lens, 4, 2, 4, 64, 16)

    actual, actual_lens = combine_topk_swa_indices(*args)
    expected, expected_lens = vllm_combine_topk_swa_indices(*args)

    fg_testing.assert_equal(actual, expected)
    fg_testing.assert_equal(actual_lens, expected_lens)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_pair_metadata_four_modes():
    """Cover no-growth/growth before and after SWA saturation."""
    device = "cuda"
    topk = 8
    topk_indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(8, 1)
    # query_len=8 and seq_len=12 make the even rows start at positions
    # 4, 6, 8, and 10. For R=4 and W=8 these are modes 1, 3, 2, and 4.
    query_start_loc = torch.tensor([17, 25], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([12], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([12], device=device, dtype=torch.int32)
    args = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        8,
        4,
        topk,
        64,
        16,
    )

    actual, actual_lens, actual_metadata = combine_topk_swa_indices(
        *args, return_pair_metadata=True
    )
    expected, expected_lens, expected_metadata = _reference_combine_with_pair_metadata(
        *args
    )

    fg_testing.assert_equal(actual, expected.to(device))
    fg_testing.assert_equal(actual_lens, expected_lens.to(device))
    fg_testing.assert_equal(actual_metadata, expected_metadata.to(device))
    assert actual_metadata.dtype == torch.int32
    # Metadata packs the first row's top-k length above the low three mode bits.
    assert actual_metadata.cpu().tolist() == [9, 11, 18, 20]


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@pytest.mark.parametrize("assume_ordered_topk", [False, True])
def test_combine_topk_swa_indices_zero_window_disables_pair_metadata(
    assume_ordered_topk,
):
    """A zero-width window has no consumer-compatible pair mode."""
    device = "cuda"
    topk = 8
    topk_indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(8, 1)
    args = (
        topk_indices,
        torch.tensor([0, 8], device=device, dtype=torch.int32),
        torch.tensor([8], device=device, dtype=torch.int32),
        torch.tensor([8], device=device, dtype=torch.int32),
        0,
        4,
        topk,
        64,
        16,
    )

    actual, actual_lens, actual_metadata = combine_topk_swa_indices(
        *args,
        return_pair_metadata=True,
        assume_ordered_topk=assume_ordered_topk,
    )
    expected, expected_lens, expected_metadata = _reference_combine_with_pair_metadata(
        *args, assume_ordered_topk=assume_ordered_topk
    )

    fg_testing.assert_equal(actual, expected.to(device))
    fg_testing.assert_equal(actual_lens, expected_lens.to(device))
    fg_testing.assert_equal(actual_metadata, expected_metadata.to(device))
    assert actual_metadata.cpu().tolist() == [0, 0, 0, 0]


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_swa_only_allows_zero_topk_and_n():
    """SWA-only callers use an empty top-k tensor and an N offset of zero."""
    device = "cuda"
    topk_indices = torch.empty((4, 0), device=device, dtype=torch.int32)
    args = (
        topk_indices,
        torch.tensor([0, 4], device=device, dtype=torch.int32),
        torch.tensor([4], device=device, dtype=torch.int32),
        torch.tensor([4], device=device, dtype=torch.int32),
        4,
        4,
        0,
        8,
        0,
    )

    actual, actual_lens, actual_metadata = combine_topk_swa_indices(
        *args, return_pair_metadata=True
    )
    expected, expected_lens, expected_metadata = _reference_combine_with_pair_metadata(
        *args
    )

    fg_testing.assert_equal(actual, expected.to(device))
    fg_testing.assert_equal(actual_lens, expected_lens.to(device))
    fg_testing.assert_equal(actual_metadata, expected_metadata.to(device))
    assert actual_lens.cpu().tolist() == [1, 2, 3, 4]
    assert actual_metadata.cpu().tolist() == [1, 1]


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_pair_metadata_request_boundaries_and_odd_tail():
    """Global pairs must not cross requests, and an odd final row stays unpaired."""
    device = "cuda"
    topk = 4
    topk_indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(9, 1)
    # Relative request ranges are [0, 3), [3, 7), and [7, 9). Thus global
    # pairs (2, 3) and (6, 7) cross requests, while row 8 is an odd tail.
    query_start_loc = torch.tensor([7, 10, 14, 16], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([3, 4, 2], device=device, dtype=torch.int32)
    gather_lens = seq_lens.clone()
    args = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        8,
        4,
        topk,
        64,
        16,
    )

    actual, actual_lens, actual_metadata = combine_topk_swa_indices(
        *args, return_pair_metadata=True
    )
    expected, expected_lens, expected_metadata = _reference_combine_with_pair_metadata(
        *args
    )

    fg_testing.assert_equal(actual, expected.to(device))
    fg_testing.assert_equal(actual_lens, expected_lens.to(device))
    fg_testing.assert_equal(actual_metadata, expected_metadata.to(device))
    assert actual_metadata.cpu().tolist() == [1, 0, 1, 0, 0]


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_pair_metadata_rejects_exact_mismatch():
    """One mismatch in the shared active top-k prefix forces fallback."""
    device = "cuda"
    topk = 8
    topk_indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(8, 1)
    # Break the compared prefix independently for mode candidates 1, 3, 2, 4.
    topk_indices[1, 0] = 91
    topk_indices[3, 0] = 92
    topk_indices[5, 1] = 93
    topk_indices[7, 0] = 94
    query_start_loc = torch.tensor([0, 8], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([12], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([12], device=device, dtype=torch.int32)
    args = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        8,
        4,
        topk,
        64,
        16,
    )

    actual, actual_lens, actual_metadata = combine_topk_swa_indices(
        *args, return_pair_metadata=True
    )
    expected, expected_lens, expected_metadata = _reference_combine_with_pair_metadata(
        *args
    )

    fg_testing.assert_equal(actual, expected.to(device))
    fg_testing.assert_equal(actual_lens, expected_lens.to(device))
    fg_testing.assert_equal(actual_metadata, expected_metadata.to(device))
    assert actual_metadata.count_nonzero().item() == 0


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_pair_metadata_trusted_ordered_topk():
    """The producer hint skips comparison only inside its short-row region."""
    device = "cuda"
    topk = 8
    topk_indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(8, 1)
    topk_indices[1, 0] = 91
    topk_indices[3, 0] = 92
    topk_indices[5, 1] = 93
    topk_indices[7, 0] = 94
    args = (
        topk_indices,
        torch.tensor([0, 8], device=device, dtype=torch.int32),
        torch.tensor([12], device=device, dtype=torch.int32),
        torch.tensor([12], device=device, dtype=torch.int32),
        8,
        4,
        topk,
        64,
        16,
    )

    _, _, exact_metadata = combine_topk_swa_indices(*args, return_pair_metadata=True)
    actual, actual_lens, trusted_metadata = combine_topk_swa_indices(
        *args,
        return_pair_metadata=True,
        assume_ordered_topk=True,
    )
    expected, expected_lens, expected_metadata = _reference_combine_with_pair_metadata(
        *args, assume_ordered_topk=True
    )

    assert exact_metadata.count_nonzero().item() == 0
    fg_testing.assert_equal(actual, expected.to(device))
    fg_testing.assert_equal(actual_lens, expected_lens.to(device))
    fg_testing.assert_equal(trusted_metadata, expected_metadata.to(device))
    assert trusted_metadata.cpu().tolist() == [9, 11, 18, 20]

    # At p=12 the raw compressed length is three, above topk=2. Even exactly
    # equal source rows must not be trusted without an explicit comparison.
    capped_indices = torch.tensor([[0, 1], [0, 1]], device=device, dtype=torch.int32)
    capped_args = (
        capped_indices,
        torch.tensor([0, 2], device=device, dtype=torch.int32),
        torch.tensor([14], device=device, dtype=torch.int32),
        torch.tensor([14], device=device, dtype=torch.int32),
        8,
        4,
        2,
        64,
        16,
    )
    _, _, capped_exact = combine_topk_swa_indices(
        *capped_args, return_pair_metadata=True
    )
    _, _, capped_trusted = combine_topk_swa_indices(
        *capped_args,
        return_pair_metadata=True,
        assume_ordered_topk=True,
    )
    assert capped_exact.cpu().tolist() == [18]
    assert capped_trusted.cpu().tolist() == [0]


def _minimal_combine_inputs():
    return (
        torch.zeros((1, 1), dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        1,
        1,
        1,
        4,
        1,
    )


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize(
    ("arg_index", "bad_value"),
    [
        (0, "not a tensor"),
        (1, "not a tensor"),
        (2, "not a tensor"),
        (3, "not a tensor"),
    ],
)
def test_combine_topk_swa_indices_rejects_non_tensor_inputs(arg_index, bad_value):
    args = list(_minimal_combine_inputs())
    args[arg_index] = bad_value
    with pytest.raises(TypeError):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("arg_index", [0, 1, 2, 3])
def test_combine_topk_swa_indices_rejects_wrong_tensor_dtype(arg_index):
    args = list(_minimal_combine_inputs())
    args[arg_index] = args[arg_index].to(torch.int64)
    with pytest.raises(ValueError, match="dtype torch.int32"):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("arg_index", [0, 1, 2, 3])
def test_combine_topk_swa_indices_rejects_wrong_tensor_rank(arg_index):
    args = list(_minimal_combine_inputs())
    args[arg_index] = args[arg_index].unsqueeze(0)
    with pytest.raises(ValueError, match="must be [12]D"):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
def test_combine_topk_swa_indices_rejects_noncontiguous_tensors():
    cases = []

    args = list(_minimal_combine_inputs())
    args[0] = torch.empty((2, 2), dtype=torch.int32)[:, ::2]
    assert not args[0].is_contiguous()
    cases.append(args)

    args = list(_minimal_combine_inputs())
    args[1] = torch.empty(4, dtype=torch.int32)[::2]
    assert not args[1].is_contiguous()
    cases.append(args)

    args = list(_minimal_combine_inputs())
    args[1] = torch.tensor([0, 1, 2], dtype=torch.int32)
    args[2] = torch.empty(4, dtype=torch.int32)[::2]
    args[3] = torch.ones(2, dtype=torch.int32)
    assert not args[2].is_contiguous()
    cases.append(args)

    args = list(_minimal_combine_inputs())
    args[1] = torch.tensor([0, 1, 2], dtype=torch.int32)
    args[2] = torch.ones(2, dtype=torch.int32)
    args[3] = torch.empty(4, dtype=torch.int32)[::2]
    assert not args[3].is_contiguous()
    cases.append(args)

    for args in cases:
        with pytest.raises(ValueError, match="must be contiguous"):
            combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("arg_index", [1, 2, 3])
def test_combine_topk_swa_indices_rejects_mixed_devices(arg_index):
    args = list(_minimal_combine_inputs())
    args[arg_index] = torch.empty_like(args[arg_index], device="meta")
    with pytest.raises(ValueError, match="same device"):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
def test_combine_topk_swa_indices_rejects_inconsistent_shapes():
    args = list(_minimal_combine_inputs())
    args[1] = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(ValueError, match=r"num_reqs \+ 1"):
        combine_topk_swa_indices(*args)

    args = list(_minimal_combine_inputs())
    args[3] = torch.empty(0, dtype=torch.int32)
    with pytest.raises(ValueError, match="same length"):
        combine_topk_swa_indices(*args)

    args = list(_minimal_combine_inputs())
    args[6] = 2
    with pytest.raises(ValueError, match="row width"):
        combine_topk_swa_indices(*args)

    args = list(_minimal_combine_inputs())
    args[1] = torch.tensor([0], dtype=torch.int32)
    args[2] = torch.empty(0, dtype=torch.int32)
    args[3] = torch.empty(0, dtype=torch.int32)
    with pytest.raises(ValueError, match="at least one request"):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("arg_index", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("bad_value", [None, True, 1.0, "1"])
def test_combine_topk_swa_indices_rejects_non_integer_parameters(arg_index, bad_value):
    args = list(_minimal_combine_inputs())
    args[arg_index] = bad_value
    with pytest.raises(TypeError, match="must be an int"):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("arg_index", [5, 7])
@pytest.mark.parametrize("bad_value", [0, -1])
def test_combine_topk_swa_indices_rejects_nonpositive_parameters(arg_index, bad_value):
    args = list(_minimal_combine_inputs())
    args[arg_index] = bad_value
    with pytest.raises(ValueError, match="must be positive"):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("arg_index", [4, 6, 8])
def test_combine_topk_swa_indices_rejects_negative_nonnegative_parameters(arg_index):
    args = list(_minimal_combine_inputs())
    args[arg_index] = -1
    with pytest.raises(ValueError, match="must be non-negative"):
        combine_topk_swa_indices(*args)


@pytest.mark.combine_topk_swa_indices
def test_combine_topk_swa_indices_empty_input_returns_initialized_empty_outputs():
    args = (
        torch.empty((0, 1), dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        0,
        1,
        1,
        1,
        1,
    )
    combined, lens, pair_metadata = combine_topk_swa_indices(
        *args, return_pair_metadata=True
    )
    assert combined.shape == (0, 128)
    assert lens.shape == (0,)
    assert pair_metadata.shape == (0,)


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("bad_value", [None, 0, 1, "true"])
def test_combine_topk_swa_indices_rejects_non_bool_return_pair_metadata(
    bad_value,
):
    with pytest.raises(TypeError):
        combine_topk_swa_indices(
            *_minimal_combine_inputs(), return_pair_metadata=bad_value
        )


@pytest.mark.combine_topk_swa_indices
@pytest.mark.parametrize("bad_value", [None, 0, 1, "true"])
def test_combine_topk_swa_indices_rejects_non_bool_assume_ordered_topk(
    bad_value,
):
    with pytest.raises(TypeError):
        combine_topk_swa_indices(
            *_minimal_combine_inputs(), assume_ordered_topk=bad_value
        )


@pytest.mark.combine_topk_swa_indices
def test_combine_topk_swa_indices_ordered_hint_requires_metadata():
    with pytest.raises(ValueError):
        combine_topk_swa_indices(*_minimal_combine_inputs(), assume_ordered_topk=True)
