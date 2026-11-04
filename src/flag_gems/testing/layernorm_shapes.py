# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# FlagGems tests/test_layer_norm.py and benchmark/test_layer_norm.py,
# commit 59840ce24ea0332a6352a595d4d9ad5ca9504572. Preserve normalized suffixes.
# The final three cases reproduce FlagGems-vllm's SkipLayerNorm reduction sizes
# at ca76e01553af9169f48a89989eb257fdcbab97aa (its input is shape[:2]).
LEGACY_LAYERNORM_CASES = (
    ((200, 36), (36,)),
    ((4096, 100), (100,)),
    ((1, 40999), (40999,)),
    ((100, 40499), (40499,)),
    ((4096, 256), (256,)),
    ((256, 512), (512,)),
    ((4096, 512), (512,)),
    ((2048, 1024), (1024,)),
    ((1024, 2048), (2048,)),
    ((4, 64), (64,)),
    ((4, 8, 8), (8, 8)),
    ((4, 4, 4, 4), (4, 4, 4)),
    ((16, 16, 64), (16, 64)),
    ((16, 16, 1024), (16, 1024)),
    ((1, 8, 4, 4), (8, 4, 4)),
    ((1, 2), (2,)),
    ((200, 40999), (40999,)),
    ((2, 32), (32,)),
)

# These existing FlagGems benchmark suffixes exceed the API's 65536 limit.
UNSUPPORTED_LEGACY_LAYERNORM_CASES = (
    ((16, 16, 4098), (16, 4098)),
    ((16, 8, 128, 128), (8, 128, 128)),
)
