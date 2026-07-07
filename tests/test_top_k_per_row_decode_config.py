from importlib import import_module


def test_top_k_per_row_decode_split_config_uses_medium_vocab_path():
    decode_op = import_module("flag_gems.fused.top_k_per_row_decode")

    assert decode_op._get_decode_split_blocks(1, 32768, 1024) == 1
    assert decode_op._get_decode_split_blocks(16, 65536, 1024) == 1
    assert decode_op._get_decode_split_blocks(1, 129280, 512) == 10
    assert decode_op._get_decode_split_blocks(16, 129280, 1024) == 10
    assert decode_op._get_decode_split_blocks(64, 129280, 1024) == 4
    assert decode_op._get_decode_split_blocks(1, 262144, 512) == 10
