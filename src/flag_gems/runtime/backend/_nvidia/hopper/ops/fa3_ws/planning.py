"""Minimal vLLM-style FA3 metadata dispatch helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FA3MetadataDispatch:
    is_paged: bool
    has_cache_kv: bool
    mode: str
    split_kv: bool
    requested_num_splits: int
    has_scheduler_metadata: bool
    metadata_source: str = "none"

    @property
    def layout(self) -> str:
        return "paged" if self.is_paged else "dense"


def fa3_tle_metadata_dispatch(
    *,
    max_query_len: int,
    is_paged: bool,
    has_cache_kv: bool,
    num_splits: int,
    has_scheduler_metadata: bool = False,
    metadata_source: str = "none",
) -> FA3MetadataDispatch:
    requested_num_splits = max(0, int(num_splits or 0))
    split_kv = has_cache_kv and requested_num_splits > 1
    if not has_cache_kv:
        mode = "prefill"
    elif split_kv:
        mode = "splitkv_decode"
    elif max_query_len <= 1:
        mode = "direct_decode"
    else:
        mode = "multi_token_decode"

    return FA3MetadataDispatch(
        is_paged=is_paged,
        has_cache_kv=has_cache_kv,
        mode=mode,
        split_kv=split_kv,
        requested_num_splits=requested_num_splits,
        has_scheduler_metadata=has_scheduler_metadata,
        metadata_source=metadata_source,
    )


__all__ = ["FA3MetadataDispatch", "fa3_tle_metadata_dispatch"]
