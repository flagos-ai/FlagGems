"""Tests for SQLPersistantModel, covering both SQLAlchemy 1.4 and >= 2.0.

These tests are device independent (sqlite backed); they run in the
sqlalchemy-compat CI workflow against both supported SQLAlchemy majors.
"""

import importlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
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


def test_blob_mode_round_trip(tmp_path):
    # More keys than key_count_limit (64 for sqlite) switches the model to
    # hash+blob columns; exercise that path on both SQLAlchemy majors too.
    model = SQLPersistantModel(f"sqlite:///{tmp_path}/cache.db")
    keys = list(range(100))
    config = triton.Config({"BLOCK_SIZE": 512}, num_warps=4)

    model.put_config("op_blob", keys, config)
    got = model.get_config("op_blob", keys)

    assert got is not None
    assert got.all_kwargs()["BLOCK_SIZE"] == 512


def test_parallel_connections_initialize_wal(tmp_path):
    url = f"sqlite:///{tmp_path}/shared.db"
    ready = Barrier(4)

    def connect():
        model = SQLPersistantModel(url)
        try:
            ready.wait(timeout=10)
            with model.engine.connect() as connection:
                assert (
                    connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
                )
                assert (
                    connection.exec_driver_sql("PRAGMA busy_timeout").scalar() == 30000
                )
                assert connection.exec_driver_sql("PRAGMA synchronous").scalar() == 1
        finally:
            model.engine.dispose()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: connect(), range(4)))


@pytest.fixture
def sqlite_connect_hook(tmp_path, monkeypatch):
    module = importlib.import_module("flag_gems.utils.models.sql")
    hooks = []

    def listens_for(engine, event):
        assert event == "connect"

        def register(callback):
            hooks.append(callback)
            return callback

        return register

    monkeypatch.setattr(module.sqlalchemy.event, "listens_for", listens_for)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    model = SQLPersistantModel(f"sqlite:///{tmp_path}/hook.db")
    assert len(hooks) == 1
    try:
        yield hooks[0]
    finally:
        model.engine.dispose()


class _PragmaCursor:
    def __init__(self, failure):
        self.failure = failure
        self.mode = "delete"
        self.closed = False
        self.synchronous = None

    def cursor(self):
        return self

    def execute(self, statement):
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        if statement == "PRAGMA journal_mode=WAL":
            self.mode = "wal"
        elif statement == "PRAGMA synchronous=NORMAL":
            self.synchronous = "normal"
        return self

    def fetchone(self):
        return (self.mode,)

    def close(self):
        self.closed = True


def test_connection_retries_transient_lock(sqlite_connect_hook):
    cursor = _PragmaCursor(sqlite3.OperationalError("database is locked"))
    sqlite_connect_hook(cursor, None)
    assert cursor.mode == "wal"
    assert cursor.synchronous == "normal"
    assert cursor.closed


def test_connection_preserves_non_lock_error(sqlite_connect_hook):
    error = sqlite3.OperationalError("disk I/O error")
    cursor = _PragmaCursor(error)
    with pytest.raises(sqlite3.OperationalError) as caught:
        sqlite_connect_hook(cursor, None)
    assert caught.value is error
    assert cursor.closed
