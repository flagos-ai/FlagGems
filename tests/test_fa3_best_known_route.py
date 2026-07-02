import pytest

from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.best_known import (
    ROUTE_CURRENT_FA3,
    ROUTE_FA2_FALLBACK,
    classify_fa3_workload,
    fa3_tle_best_route_mode,
    select_fa3_best_route,
)


def test_fa3_best_route_env_validation(monkeypatch):
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "bad")
    with pytest.raises(RuntimeError, match="FLAG_GEMS_FA3_TLE_BEST_ROUTE"):
        fa3_tle_best_route_mode()


def test_fa3_best_route_default_is_non_fallback(monkeypatch):
    monkeypatch.delenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", raising=False)
    assert fa3_tle_best_route_mode() == "fa3_only"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            dict(
                total_q=8192,
                batch_size=4,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                is_paged=False,
            ),
            "dense_prefill_or_long",
        ),
        (
            dict(
                total_q=16,
                batch_size=16,
                max_seqlen_q=1,
                max_seqlen_k=1024,
                is_paged=True,
            ),
            "paged_decode",
        ),
        (
            dict(
                total_q=4096,
                batch_size=32,
                max_seqlen_q=2048,
                max_seqlen_k=4096,
                is_paged=True,
            ),
            "paged_serve_mixed",
        ),
        (
            dict(
                total_q=16384,
                batch_size=4,
                max_seqlen_q=4096,
                max_seqlen_k=4096,
                is_paged=True,
            ),
            "paged_uniform_prefill",
        ),
    ],
)
def test_fa3_best_workload_classification(kwargs, expected):
    assert classify_fa3_workload(**kwargs) == expected


def test_fa3_best_route_default_keeps_fa3():
    kwargs = dict(
        total_q=16,
        batch_size=16,
        max_seqlen_q=1,
        max_seqlen_k=1024,
        is_paged=True,
    )
    route = select_fa3_best_route(force_family_id=-1, **kwargs)
    assert route.route == ROUTE_CURRENT_FA3


@pytest.mark.parametrize(
    "kwargs,expected_route",
    [
        (
            dict(
                total_q=8192,
                batch_size=4,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                is_paged=False,
            ),
            ROUTE_CURRENT_FA3,
        ),
        (
            dict(
                total_q=16,
                batch_size=16,
                max_seqlen_q=1,
                max_seqlen_k=1024,
                is_paged=True,
            ),
            ROUTE_FA2_FALLBACK,
        ),
    ],
)
def test_fa3_best_route_legacy_auto(monkeypatch, kwargs, expected_route):
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "auto")
    route = select_fa3_best_route(force_family_id=-1, **kwargs)
    assert route.route == expected_route


def test_fa3_best_route_force_family_keeps_fa3():
    route = select_fa3_best_route(
        total_q=16,
        batch_size=16,
        max_seqlen_q=1,
        max_seqlen_k=1024,
        is_paged=True,
        force_family_id=5,
    )
    assert route.route == ROUTE_CURRENT_FA3


def test_fa3_best_route_env_overrides(monkeypatch):
    kwargs = dict(
        total_q=8192,
        batch_size=4,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        is_paged=False,
        force_family_id=-1,
    )
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "fa2_only")
    assert select_fa3_best_route(**kwargs).route == ROUTE_FA2_FALLBACK
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "fa3_only")
    assert select_fa3_best_route(**kwargs).route == ROUTE_CURRENT_FA3
