import csv
import importlib.util
import sqlite3
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "flagtune"
    / "tuning_latency_compare.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("tuning_latency_compare", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_benchmark_output_extracts_mm_shape_and_metrics():
    mod = load_module()
    text = """
SUCCESS               0.098896            0.097024               1.019              20.966          [torch.Size([8, 4096]), torch.Size([4096, 31040])]
"""

    records = mod.parse_benchmark_output(text)

    assert len(records) == 1
    record = records[0]
    assert record.status == "SUCCESS"
    assert record.shape_key == "1,8,31040,4096"
    assert record.torch_ms == 0.098896
    assert record.gems_ms == 0.097024
    assert record.speedup == 1.019
    assert record.tflops == 20.966


def test_method_env_isolates_flaggems_cache_and_legacy_flagtune(monkeypatch, tmp_path):
    mod = load_module()
    monkeypatch.setenv("USE_FLAGTUNE", "1")
    monkeypatch.setenv("FLAGTUNE_INCLUDE", "mm")

    default_env = mod.method_env("default", tmp_path / "default")
    expanded_env = mod.method_env("expanded", tmp_path / "expanded")

    assert default_env["FLAGGEMS_CACHE_DIR"] == str(tmp_path / "default")
    assert "USE_FLAGTUNE" not in default_env
    assert "FLAGTUNE_INCLUDE" not in default_env
    assert expanded_env["FLAGGEMS_CACHE_DIR"] == str(tmp_path / "expanded")
    assert expanded_env["USE_FLAGTUNE"] == "1"
    assert "FLAGTUNE_INCLUDE" not in expanded_env


def test_merge_method_rows_reports_cold_and_hot_metrics(tmp_path):
    mod = load_module()
    shape = mod.ShapeRecord((1, 8, 31040, 4096), count=7)
    cold = mod.ColdRecord(
        status="SUCCESS",
        shape_key=shape.shape_key,
        cold_gems_ms=2.0,
        cold_torch_ms=None,
    )
    hot = mod.BenchRecord(
        status="SUCCESS",
        torch_ms=0.1,
        gems_ms=0.5,
        speedup=0.2,
        tflops=4.0,
        shape_text="[torch.Size([8, 4096]), torch.Size([4096, 31040])]",
        shape_key=shape.shape_key,
    )
    cold_result = mod.ColdRunResult(
        returncode=0,
        records=[cold],
        log_path=tmp_path / "cold.log",
        jsonl_path=tmp_path / "cold.jsonl",
        cold_pass_wall_s=12.5,
        cache_summary=mod.CacheSummary(
            config_cache_db_bytes=4096,
            benchmark_cache_rows=99,
        ),
    )
    hot_completed = type("Completed", (), {"returncode": 0})()

    rows = mod.merge_method_rows(
        "default",
        [shape],
        [cold],
        [hot],
        tmp_path / "cache",
        cold_result,
        hot_completed,
    )

    assert rows == [
        {
            "shape": "1, 8, 31040, 4096",
            "shape_key": "1,8,31040,4096",
            "count": 7,
            "method": "default",
            "status": "ok",
            "cold_gems_ms": 2.0,
            "hot_gems_ms": 0.5,
            "cold_torch_ms": None,
            "torch_ms": 0.1,
            "speedup": 0.2,
            "cold_hot_ratio": 4.0,
            "cache_dir": str(tmp_path / "cache"),
            "cold_wall_source": "first_call_wall_clock",
            "hot_latency_source": "pytest_do_bench",
            "cold_pass_wall_s": 12.5,
            "config_cache_db_bytes": 4096,
            "benchmark_cache_rows": 99,
            "error": "",
        }
    ]


def test_parse_cold_worker_output_reads_json_lines():
    mod = load_module()
    text = """
noise
{"shape_key": "1,8,31040,4096", "status": "SUCCESS", "cold_gems_ms": 123.0, "cold_torch_ms": null, "error": ""}
"""

    records = mod.parse_cold_worker_output(text)

    assert records == [
        mod.ColdRecord(
            status="SUCCESS",
            shape_key="1,8,31040,4096",
            cold_gems_ms=123.0,
            cold_torch_ms=None,
            error="",
        )
    ]


def test_split_shape_records_balances_by_flops():
    mod = load_module()
    shapes = [
        mod.ShapeRecord((1, 1, 1, 1)),
        mod.ShapeRecord((1, 100, 100, 100)),
        mod.ShapeRecord((1, 2, 2, 2)),
    ]

    chunks = mod.split_shape_records(shapes, parallel=2)

    assert len(chunks) == 2
    assert sorted(record.shape_key for chunk in chunks for record in chunk) == sorted(
        record.shape_key for record in shapes
    )


def test_summarize_config_cache_counts_benchmark_rows(tmp_path):
    mod = load_module()
    config_cache = tmp_path / "config_cache"
    config_cache.mkdir()
    db_path = config_cache / "TunedConfig_test.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute('create table "kernel_benchmark-abc" (id integer primary key)')
        con.execute('create table "kernel_config-abc" (id integer primary key)')
        con.executemany(
            'insert into "kernel_benchmark-abc" (id) values (?)',
            [(1,), (2,), (3,)],
        )
        con.commit()
    finally:
        con.close()

    summary = mod.summarize_config_cache(tmp_path)

    assert summary.config_cache_db_bytes > 0
    assert summary.benchmark_cache_rows == 3


def test_write_by_shape_csv_matches_default_and_expanded_semantics(tmp_path):
    mod = load_module()
    common = {
        "shape": "1, 8, 31040, 4096",
        "shape_key": "1,8,31040,4096",
        "count": 7,
        "status": "ok",
        "cold_wall_source": "first_call_wall_clock",
        "hot_latency_source": "pytest_do_bench",
    }
    rows = [
        {
            **common,
            "method": "default",
            "cold_gems_ms": 2.0,
            "hot_gems_ms": 0.5,
            "torch_ms": 0.4,
        },
        {
            **common,
            "method": "expanded",
            "cold_gems_ms": 10.0,
            "hot_gems_ms": 0.45,
            "torch_ms": 0.41,
        },
    ]
    output = tmp_path / "tuning_latency_compare_by_shape.csv"

    mod.write_by_shape_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))
    assert len(result) == 1
    assert result[0]["shape"] == common["shape"]
    assert result[0]["shape_key"] == common["shape_key"]
    assert float(result[0]["default_tuning_ms"]) == 2.0
    assert float(result[0]["expanded_tuning_ms"]) == 10.0
    assert float(result[0]["default_vs_expanded_tuning_speedup"]) == 5.0
    assert float(result[0]["default_perf_pct_of_expanded_hot"]) == 90.0
