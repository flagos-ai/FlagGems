# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# GCU400: override fused_recurrent with the in-place final-state aliasing fix
# (read the initial state from a snapshot). chunk_gated_delta_rule reuses the
# generic implementation.
from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd

__all__ = [
    "fused_recurrent_gated_delta_rule_fwd",
]
