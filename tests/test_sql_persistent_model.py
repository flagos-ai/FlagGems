"""Tests for SQLPersistantModel, covering both SQLAlchemy 1.4 and >= 2.0.

These tests are device independent (sqlite backed); they run in the
sqlalchemy-compat CI workflow against both supported SQLAlchemy majors.
"""

import triton

from flag_gems.utils.models.sql import SQLPersistantModel


def test_config_round_trip(tmp_path):
    model = SQLPersistantModel(f"sqlite:///{tmp_path}/cache.db")
    keys = [128, "float16", True]
    config = triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2)

    model.put_config("op_config", keys, config)
    got = model.get_config("op_config", keys)

    assert got is not None
    assert got.all_kwargs()["BLOCK_SIZE"] == 1024
    assert got.all_kwargs()["num_warps"] == 4


def test_benchmark_round_trip(tmp_path):
    model = SQLPersistantModel(f"sqlite:///{tmp_path}/cache.db")
    keys = [128, "float16", True]
    config = triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2)

    model.put_benchmark("op_bench", keys, config, (1.5, 1.0, 2.0))

    assert model.get_benchmark("op_bench", keys, config) == (1.5, 1.0, 2.0)


def test_reload_existing_db_via_automap(tmp_path):
    db_url = f"sqlite:///{tmp_path}/cache.db"
    keys = [64, "float32"]
    config = triton.Config({"BLOCK": 256}, num_warps=8)
    SQLPersistantModel(db_url).put_config("op_reload", keys, config)

    # a fresh engine on the same db exercises build_sql_model_by_db/automap
    reloaded = SQLPersistantModel(db_url).get_config("op_reload", keys)

    assert reloaded is not None
    assert reloaded.all_kwargs()["BLOCK"] == 256
