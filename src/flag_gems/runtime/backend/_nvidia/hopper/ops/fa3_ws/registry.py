"""Registry for experimental FA3 warp-specialized variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WSVariant:
    name: str
    force_path: str
    kernel_module: str
    persistent: bool
    sync_mode: str
    paged: bool
    shape_kind: str
    description: str


_VARIANTS: dict[str, WSVariant] = {
    "ws_simple_dense_decode": WSVariant(
        name="ws_simple_dense_decode",
        force_path="ws_simple",
        kernel_module="fa_hopper_persistent_pingpong",
        persistent=True,
        sync_mode="pingpong",
        paged=False,
        shape_kind="decode",
        description="Persistent ping-pong WS, dense decode shape.",
    ),
    "ws_simple_paged_decode": WSVariant(
        name="ws_simple_paged_decode",
        force_path="ws_simple",
        kernel_module="fa_hopper_persistent_pingpong",
        persistent=True,
        sync_mode="pingpong",
        paged=True,
        shape_kind="paged_decode",
        description="Persistent ping-pong WS, paged decode gather shape.",
    ),
    "ws_simple_small_dense": WSVariant(
        name="ws_simple_small_dense",
        force_path="ws_simple",
        kernel_module="fa_hopper_persistent_pingpong",
        persistent=True,
        sync_mode="pingpong",
        paged=False,
        shape_kind="small",
        description="Persistent ping-pong WS, small dense shape.",
    ),
    "ws_short_dense_decode": WSVariant(
        name="ws_short_dense_decode",
        force_path="ws_short",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="sync",
        paged=False,
        shape_kind="decode",
        description="Nonpersistent TLX-style WS, dense decode shape.",
    ),
    "ws_short_paged_decode": WSVariant(
        name="ws_short_paged_decode",
        force_path="ws_short",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="sync",
        paged=True,
        shape_kind="paged_decode",
        description="Nonpersistent TLX-style WS, paged decode gather shape.",
    ),
    "ws_short_small_dense": WSVariant(
        name="ws_short_small_dense",
        force_path="ws_short",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="sync",
        paged=False,
        shape_kind="small",
        description="Nonpersistent TLX-style WS, small dense shape.",
    ),
    "ws_sync_decode": WSVariant(
        name="ws_sync_decode",
        force_path="ws_sync_decode",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="sync",
        paged=False,
        shape_kind="decode",
        description="Explicit sync candidate, dense decode shape.",
    ),
    "ws_sync_small": WSVariant(
        name="ws_sync_small",
        force_path="ws_sync_small",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="sync",
        paged=False,
        shape_kind="small",
        description="Explicit sync candidate, small dense shape.",
    ),
    "ws_sync_paged_decode": WSVariant(
        name="ws_sync_paged_decode",
        force_path="ws_sync_paged_decode",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="sync",
        paged=True,
        shape_kind="paged_decode",
        description="Explicit sync candidate, paged decode gather shape.",
    ),
    "ws_pipe2_decode": WSVariant(
        name="ws_pipe2_decode",
        force_path="ws_pipe2_decode",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="pipe2",
        paged=False,
        shape_kind="decode",
        description="Two-stage pipeline candidate, dense decode shape.",
    ),
    "ws_pipe2_paged_decode": WSVariant(
        name="ws_pipe2_paged_decode",
        force_path="ws_pipe2_paged_decode",
        kernel_module="fa_hopper_nonpersistent_tlx_style",
        persistent=False,
        sync_mode="pipe2",
        paged=True,
        shape_kind="paged_decode",
        description="Two-stage pipeline candidate, paged decode gather shape.",
    ),
}

_ALIASES: dict[str, tuple[str, ...]] = {
    "all": tuple(_VARIANTS),
    "persistent": tuple(name for name, spec in _VARIANTS.items() if spec.persistent),
    "nonpersistent": tuple(name for name, spec in _VARIANTS.items() if not spec.persistent),
    "paged": tuple(name for name, spec in _VARIANTS.items() if spec.paged),
    "dense": tuple(name for name, spec in _VARIANTS.items() if not spec.paged),
    "ws_simple": (
        "ws_simple_dense_decode",
        "ws_simple_paged_decode",
        "ws_simple_small_dense",
    ),
    "ws_short": (
        "ws_short_dense_decode",
        "ws_short_paged_decode",
        "ws_short_small_dense",
    ),
}


def variant_names() -> tuple[str, ...]:
    return tuple(_VARIANTS)


def iter_variants() -> Iterable[WSVariant]:
    return _VARIANTS.values()


def get_variant(name: str) -> WSVariant:
    try:
        return _VARIANTS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted((*_VARIANTS, *_ALIASES)))
        raise KeyError(f"unknown FA3 WS variant {name!r}; expected one of {allowed}") from exc


def resolve_variant_names(selection: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(selection, str):
        selections = [selection]
    else:
        selections = list(selection)

    resolved: list[str] = []
    for item in selections:
        key = item.strip()
        if not key:
            continue
        names = _ALIASES.get(key, (key,))
        for name in names:
            if name not in _VARIANTS:
                get_variant(name)
            if name not in resolved:
                resolved.append(name)
    return tuple(resolved)
