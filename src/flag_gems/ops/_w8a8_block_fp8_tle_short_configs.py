# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite short-K schedules with exact-byte and synchronization checks."""

_H100_TLE_SHORT_CONFIGS = {
    (12, 5120, 288): {
        "max_registers": 128,
        "n_major": True,
        "tile_m": 16,
        "tile_n": 64,
    },
    (24, 5120, 288): {
        "max_registers": 192,
        "n_major": True,
        "tile_m": 32,
        "tile_n": 64,
    },
    (40, 5120, 288): {
        "max_registers": 192,
        "n_major": False,
        "tile_m": 16,
        "tile_n": 128,
    },
    (48, 5120, 288): {
        "max_registers": 192,
        "n_major": False,
        "tile_m": 16,
        "tile_n": 128,
    },
    (72, 5120, 288): {
        "max_registers": 192,
        "n_major": False,
        "tile_m": 32,
        "tile_n": 128,
    },
    (80, 5120, 288): {
        "max_registers": 192,
        "n_major": True,
        "tile_m": 32,
        "tile_n": 64,
    },
    (96, 5120, 288): {
        "max_registers": 192,
        "n_major": False,
        "tile_m": 32,
        "tile_n": 128,
    },
}
