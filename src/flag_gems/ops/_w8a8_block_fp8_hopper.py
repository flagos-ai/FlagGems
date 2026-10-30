# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fp8_hopper_static.py, revision
# 4cf6966fef013bccd7bbe4274df5132d7877b103 (Apache-2.0).
# Keep each K32 dot and its independent scale application intact.
"""Hopper block FP8 GEMM and split-K reduction kernels."""

from ._w8a8_block_fp8_sm90 import _reduce_block_fp8_split_k as _reduce_split_k
from ._w8a8_block_fp8_sm90 import _w8a8_block_fp8_matmul_hopper

__all__ = ["_reduce_split_k", "_w8a8_block_fp8_matmul_hopper"]
