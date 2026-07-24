"""Read Pretune shape YAML and write stable run artifacts.

Inputs:
    Shape input is a required PyYAML document whose selected top-level operator
    contains ``shape_spec`` or legacy ``shape_desc`` and a ``shapes`` list.
    Output helpers accept ordered result mappings produced by benchmark workers.

Outputs:
    The module writes ``pretune.csv``, ``pretune.jsonl``, ``manifest.json``, and
    the combined ``pretune.log`` in a timestamped run directory.

Implementation:
    YAML parsing is intentionally operator agnostic: declared field names and
    row values are preserved and handed to the compiled data-driven shape
    schema for semantic validation. JSONL uses grouped Schema v2 objects, while
    CSV writes the equivalent dimensions and metrics as stable flat columns.

Limitations:
    PyYAML is mandatory.  YAML aliases and values are accepted according to
    ``yaml.safe_load`` behavior, but each shape must be a flat sequence with the
    exact declared length.  Artifact writes are not atomic and do not resume a
    partially written run.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from flag_gems.utils.flagtune.output_schema import (
    pretune_csv_fieldnames,
    pretune_csv_row,
    pretune_json_row,
)


class PretuneIOError(RuntimeError):
    """Report invalid shape input or an artifact I/O contract violation."""


@dataclass(frozen=True)
class ShapeConfig:
    """Store one operator's generic, unvalidated YAML shape table.

    Attributes:
        operator_name: Exact selected top-level YAML key.
        shape_spec: Declared fields in source order.
        rows: Raw row tuples whose lengths match ``shape_spec``.
    """

    operator_name: str
    shape_spec: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


def _parse_shape_spec(raw: Any, operator_name: str) -> tuple[str, ...]:
    """Normalize list or comma-separated shape fields without renaming them."""
    if isinstance(raw, str):
        fields = tuple(field.strip() for field in raw.split(","))
    elif isinstance(raw, (list, tuple)):
        fields = tuple(str(field).strip() for field in raw)
    else:
        raise PretuneIOError(
            f"{operator_name}.shape_spec/shape_desc must be a list or "
            "comma-separated string"
        )
    if not fields or any(not field for field in fields):
        raise PretuneIOError(
            f"{operator_name} shape specification contains an empty field"
        )
    return fields


def load_shape_config(path: Path, operator_name: str) -> ShapeConfig:
    """Load one operator entry without applying operator-specific semantics.

    Args:
        path: YAML file to read with ``yaml.safe_load``.
        operator_name: Exact top-level key to select.

    Returns:
        A :class:`ShapeConfig` preserving declared field and row order.

    Raises:
        PretuneIOError: If PyYAML is unavailable, the file or selected entry is
            malformed, or a row length differs from the shape specification.

    Notes:
        Dimension names, types, positivity, ``Count``, and variant constraints
        are intentionally left to the compiled operator YAML schema.
    """

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise PretuneIOError(
            "PyYAML is required; install the FlagGems dependencies"
        ) from exc

    if not path.is_file():
        raise PretuneIOError(f"shape config does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise PretuneIOError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PretuneIOError("shape config root must be a mapping")
    if operator_name not in payload:
        raise PretuneIOError(
            f"shape config does not contain exact operator key {operator_name!r}"
        )
    entry = payload[operator_name]
    if not isinstance(entry, Mapping):
        raise PretuneIOError(f"shape config entry {operator_name!r} must be a mapping")
    raw_spec = entry.get("shape_spec", entry.get("shape_desc"))
    if raw_spec is None:
        raise PretuneIOError(
            f"{operator_name} must define shape_spec or legacy shape_desc"
        )
    shape_spec = _parse_shape_spec(raw_spec, operator_name)
    raw_shapes = entry.get("shapes")
    if not isinstance(raw_shapes, list):
        raise PretuneIOError(f"{operator_name}.shapes must be a list")
    rows = []
    for index, row in enumerate(raw_shapes):
        if not isinstance(row, (list, tuple)):
            raise PretuneIOError(f"{operator_name}.shapes[{index}] must be a list")
        if len(row) != len(shape_spec):
            raise PretuneIOError(
                f"{operator_name}.shapes[{index}] has {len(row)} values but "
                f"the shape specification has {len(shape_spec)} fields"
            )
        rows.append(tuple(row))
    if not rows:
        raise PretuneIOError(f"shape config contains no shapes for {operator_name!r}")
    return ShapeConfig(operator_name, shape_spec, tuple(rows))


def make_run_dir(output_root: Path, op_text: str) -> Path:
    """Create and return a collision-resistant timestamped output directory.

    ``op_text`` is made path-safe only by replacing ``/`` with ``-``; callers
    must provide an already validated operator expression.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{timestamp}_{op_text.replace('/', '-')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def remove_intermediate_artifacts(
    run_dir: Path, paths: Sequence[Path]
) -> list[str]:
    """Delete explicitly named process artifacts below one completed run.

    Args:
        run_dir: Existing timestamped run directory that must remain intact.
        paths: Files or directories produced only to complete the run.

    Returns:
        Removed paths relative to ``run_dir`` in deterministic order. Missing
        paths are ignored so successful shard merging and cleanup compose.

    Raises:
        PretuneIOError: If any requested path resolves to the run directory
            itself, escapes it, or cannot be removed.

    Safety:
        Every target is resolved and validated before the first deletion. This
        helper cannot remove the run root, an external database, or a sibling
        output directory even if a caller supplies such a path accidentally.
    """
    try:
        root = run_dir.resolve(strict=True)
    except OSError as exc:
        raise PretuneIOError(f"cannot resolve run directory {run_dir}: {exc}") from exc

    validated: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path).resolve(strict=False)
        if candidate == root or root not in candidate.parents:
            raise PretuneIOError(
                f"refusing to remove non-intermediate path outside run: {raw_path}"
            )
        validated.append(candidate)

    removed: list[str] = []
    for candidate in sorted(set(validated), key=lambda item: item.parts):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        relative = str(candidate.relative_to(root))
        try:
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
        except OSError as exc:
            raise PretuneIOError(
                f"cannot remove intermediate artifact {candidate}: {exc}"
            ) from exc
        removed.append(relative)
    return removed


def write_outputs(
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    shape_fields: Sequence[str],
) -> None:
    """Write ordered result rows to JSONL and CSV.

    Args:
        run_dir: Existing run directory that receives both files.
        rows: Worker results in final selection order.
        shape_fields: Operator-defined dimension columns inserted after the
            operator identity columns.

    Outputs:
        ``pretune.jsonl`` preserves structured values.  ``pretune.csv`` uses a
        fixed reporting schema and compact JSON for nested shape/config values.

    Limitations:
        Private worker keys are converted to public Schema v2 names. Missing
        values become JSON nulls or empty CSV fields.
    """
    with (run_dir / "pretune.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    pretune_json_row(row, shape_fields),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")

    fieldnames = pretune_csv_fieldnames(shape_fields)
    with (run_dir / "pretune.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_row in rows:
            row = pretune_csv_row(source_row, shape_fields)
            row["input_dtypes"] = json.dumps(
                row.get("input_dtypes"), separators=(",", ":")
            )
            row["output_dtypes"] = json.dumps(
                row.get("output_dtypes"), separators=(",", ":")
            )
            row["best_config"] = json.dumps(
                row.get("best_config"), sort_keys=True, separators=(",", ":")
            )
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a human-readable, key-sorted JSON run manifest.

    Args:
        path: Destination file inside the caller-managed run directory.
        payload: JSON-compatible planning, environment, worker, and artifact
            metadata assembled by Pretune.

    Returns:
        ``None``.  The destination is created or replaced with UTF-8 JSON.

    Usage:
        Pretune calls this once after worker completion so the manifest records
        final return codes, database-merge status, and artifact paths.

    Limitations:
        The write is not atomic and this helper does not create parent
        directories or coerce non-JSON-compatible objects.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def combine_logs(
    run_dir: Path, worker_logs: Sequence[Path], header: Sequence[str]
) -> None:
    """Combine run metadata and per-worker logs in deterministic worker order.

    Missing worker logs are represented by their section header and otherwise
    ignored; this helper does not interpret or truncate worker output.
    """
    with (run_dir / "pretune.log").open("w", encoding="utf-8") as output:
        for line in header:
            output.write(line)
            output.write("\n")
        for worker_id, path in enumerate(worker_logs):
            output.write(f"\n===== worker {worker_id} =====\n")
            if path.exists():
                output.write(path.read_text(encoding="utf-8"))
