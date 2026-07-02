from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.planning import (
    fa3_tle_metadata_dispatch,
)


def test_metadata_dispatch_dense_prefill():
    route = fa3_tle_metadata_dispatch(
        max_query_len=128,
        is_paged=False,
        has_cache_kv=False,
        num_splits=0,
    )
    assert route.layout == "dense"
    assert route.mode == "prefill"
    assert not route.split_kv


def test_metadata_dispatch_paged_decode():
    route = fa3_tle_metadata_dispatch(
        max_query_len=1,
        is_paged=True,
        has_cache_kv=True,
        num_splits=0,
    )
    assert route.layout == "paged"
    assert route.mode == "direct_decode"
    assert not route.split_kv


def test_metadata_dispatch_multi_token_decode_splitkv():
    route = fa3_tle_metadata_dispatch(
        max_query_len=4,
        is_paged=True,
        has_cache_kv=True,
        num_splits=4,
        has_scheduler_metadata=True,
    )
    assert route.layout == "paged"
    assert route.mode == "splitkv_decode"
    assert route.split_kv
    assert route.requested_num_splits == 4
    assert route.has_scheduler_metadata
    assert route.metadata_source == "none"


def test_metadata_dispatch_splitkv_requires_more_than_one_split():
    one_split = fa3_tle_metadata_dispatch(
        max_query_len=1,
        is_paged=True,
        has_cache_kv=True,
        num_splits=1,
    )
    many_splits = fa3_tle_metadata_dispatch(
        max_query_len=1,
        is_paged=True,
        has_cache_kv=True,
        num_splits=2,
    )
    assert not one_split.split_kv
    assert many_splits.split_kv
