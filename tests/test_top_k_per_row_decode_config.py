from importlib import import_module


def test_top_k_per_row_decode_split_config_uses_medium_vocab_path():
    decode_op = import_module("flag_gems.fused.top_k_per_row_decode")

    # Contiguous rows (stride1 == 1): medium-vocab split heuristic applies.
    assert decode_op._get_decode_split_blocks(1, 32768, 1024, 1) == 1
    assert decode_op._get_decode_split_blocks(16, 65536, 1024, 1) == 1
    assert decode_op._get_decode_split_blocks(1, 129280, 512, 1) == 10
    assert decode_op._get_decode_split_blocks(16, 129280, 1024, 1) == 10
    assert decode_op._get_decode_split_blocks(64, 129280, 1024, 1) == 4
    assert decode_op._get_decode_split_blocks(1, 262144, 512, 1) == 10


def test_top_k_per_row_decode_split_config_skips_medium_vocab_for_strided():
    decode_op = import_module("flag_gems.fused.top_k_per_row_decode")

    # Strided rows (stride1 != 1) must stay on the single-block path for the
    # medium-vocab range, since the multi-block kernel mis-addresses strided
    # rows past the first block. The >= SPLIT_WORK_THRESHOLD range keeps its
    # pre-existing behavior and is intentionally not guarded here.
    assert decode_op._get_decode_split_blocks(16, 129280, 1024, 2) == 1
    assert decode_op._get_decode_split_blocks(64, 129280, 1024, 2) == 1
    assert decode_op._get_decode_split_blocks(1, 129280, 512, 8) == 1
