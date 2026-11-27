"""Tests for SQLPersistantModel, covering both SQLAlchemy 1.4 and >= 2.0.

These tests are device independent (sqlite backed); they run in the
sqlalchemy-compat CI workflow against both supported SQLAlchemy majors.
"""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import triton

import flag_gems.utils.models.sql as sql_module
from flag_gems.utils.models.sql import SQLPersistantModel


@pytest.mark.parametrize("key_count_limit", [1, 64])
@pytest.mark.parametrize("use_async", [False, True])
def test_config_round_trip(tmp_path, key_count_limit, use_async):
    model = SQLPersistantModel(f"sqlite:///{tmp_path}/cache.db")
    model.key_count_limit = key_count_limit
    keys = [128, "float16", True]
    config = triton.Config(
        {"BLOCK_SIZE": 1024, "USE_ASYNC": use_async}, num_warps=4, num_stages=2
    )
    name = f"op_config_{key_count_limit}_{int(use_async)}"

    model.put_config(name, keys, config)
    got = model.get_config(name, keys)

    assert got is not None
    assert got.all_kwargs()["BLOCK_SIZE"] == 1024
    assert got.all_kwargs()["num_warps"] == 4
    assert got.kwargs["USE_ASYNC"] == use_async


@pytest.mark.parametrize("key_count_limit", [1, 64])
def test_benchmark_round_trip(tmp_path, key_count_limit):
    model = SQLPersistantModel(f"sqlite:///{tmp_path}/cache.db")
    model.key_count_limit = key_count_limit
    keys = [128, "float16", True]
    config = triton.Config(
        {"BLOCK_SIZE": 1024, "USE_ASYNC": True}, num_warps=4, num_stages=2
    )
    sync_config = triton.Config(
        {"BLOCK_SIZE": 1024, "USE_ASYNC": False}, num_warps=4, num_stages=2
    )
    name = f"op_bench_{key_count_limit}"

    model.put_benchmark(name, keys, config, (1.5, 1.0, 2.0))
    model.put_benchmark(name, keys, sync_config, (2.5, 2.0, 3.0))

    assert model.get_benchmark(name, keys, config) == (1.5, 1.0, 2.0)
    assert model.get_benchmark(name, keys, sync_config) == (2.5, 2.0, 3.0)


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
def locked_sqlite_db(tmp_path):
    path = tmp_path / "locked.db"
    with closing(sqlite3.connect(path)) as reader:
        reader.execute("CREATE TABLE sample (id INTEGER)")
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM sample").fetchall()
        with closing(sqlite3.connect(path, timeout=0.05)) as connection:
            yield reader, connection


def test_connection_retries_transient_lock(locked_sqlite_db, monkeypatch):
    reader, connection = locked_sqlite_db
    sleeps = []

    def release_lock(delay):
        sleeps.append(delay)
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (0,)
        reader.rollback()

    monkeypatch.setattr(
        sql_module,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=release_lock),
    )
    sql_module._configure_sqlite_connection(connection, None)
    assert len(sleeps) == 1
    assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert connection.execute("PRAGMA synchronous").fetchone() == (1,)
    assert connection.execute("PRAGMA busy_timeout").fetchone() == (50,)


def test_connection_stops_retrying_at_deadline(locked_sqlite_db, monkeypatch):
    _, connection = locked_sqlite_db
    now = 0.0

    def advance_clock(delay):
        nonlocal now
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (0,)
        now += delay

    monkeypatch.setattr(
        sql_module, "time", SimpleNamespace(monotonic=lambda: now, sleep=advance_clock)
    )
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        sql_module._configure_sqlite_connection(connection, None)
    assert now == pytest.approx(2.0)
    assert connection.execute("PRAGMA busy_timeout").fetchone() == (50,)


@pytest.fixture
def stubbed_journal_mode_change():
    responses = {
        "PRAGMA busy_timeout": (30000,),
        "PRAGMA journal_mode": ("delete",),
        "PRAGMA database_list": (0, "main", "cache.db"),
    }

    def execute(statement):
        result = responses.get(statement)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(fetchone=lambda: result)

    cursor = Mock()
    cursor.execute.side_effect = execute
    return cursor, responses


def test_connection_preserves_non_lock_error(stubbed_journal_mode_change):
    cursor, responses = stubbed_journal_mode_change
    error = sqlite3.OperationalError("disk I/O error")
    responses["PRAGMA journal_mode=WAL"] = error
    with pytest.raises(sqlite3.OperationalError) as caught:
        sql_module._configure_sqlite_connection(
            SimpleNamespace(cursor=lambda: cursor), None
        )
    assert caught.value is error
    cursor.close.assert_called_once()


def test_connection_rejects_file_database_without_wal(stubbed_journal_mode_change):
    cursor, responses = stubbed_journal_mode_change
    responses["PRAGMA journal_mode=WAL"] = ("delete",)
    with pytest.raises(sqlite3.OperationalError, match=r"journal_mode=delete"):
        sql_module._configure_sqlite_connection(
            SimpleNamespace(cursor=lambda: cursor), None
        )
    cursor.execute.assert_called_with("PRAGMA busy_timeout=30000")
    cursor.close.assert_called_once()


@pytest.mark.parametrize(
    "db_url", ["sqlite:///:memory:", "sqlite:///file:cache?mode=memory&uri=true"]
)
def test_connection_preserves_memory_journal_mode(db_url):
    model = SQLPersistantModel(db_url)
    try:
        with model.engine.connect() as connection:
            assert (
                connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "memory"
            )
            assert connection.exec_driver_sql("PRAGMA synchronous").scalar() == 1
    finally:
        model.engine.dispose()
