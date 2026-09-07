# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assemble FlagGems-trained child archives into a platform package.

Training exports one ``model.tar.gz`` per operator variant and dtype identity.
Runtime loading, however, selects one outer platform package.  This module is
the generic bridge between those two artifacts; it does not fit, rewrite, or
extract model archives.  An optional existing platform package can be supplied
to preserve models for other operators while replacing duplicate identities
with newly trained children.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from triton.flagtune.contract.archive import (
    ModelArchiveError,
    platform_package_name,
    read_model_archive,
    read_model_archive_bytes,
    read_platform_package,
    write_platform_package,
)
from triton.flagtune.contract.identity import ModelIdentity, ModelIdentityError
from triton.flagtune.contract.operator_schema import (
    load_model_config_bytes,
    model_identity_from_config,
)


class PackageAssemblyError(ValueError):
    """Report inconsistent child archives or platform-package inputs."""


def _parse_identity(value: ModelIdentity | str, location: str) -> ModelIdentity:
    """Parse one canonical platform/op/variant/dtype artifact key."""
    if isinstance(value, ModelIdentity):
        return value
    if not isinstance(value, str):
        raise PackageAssemblyError(f"{location} must be a model identity string")
    parts = value.split("/")
    if len(parts) < 5:
        raise PackageAssemblyError(
            f"{location} must be platform/op_id/variant/dtype_key: {value!r}"
        )
    try:
        identity = ModelIdentity(
            parts[0],
            "/".join(parts[1:-2]),
            parts[-2],
            parts[-1],
        )
    except (ModelIdentityError, TypeError) as exc:
        raise PackageAssemblyError(f"invalid {location}: {value!r}: {exc}") from exc
    if value != identity.artifact_key:
        raise PackageAssemblyError(
            f"{location} must be canonical: {value!r} != {identity.artifact_key!r}"
        )
    return identity


def _identity_set(
    values: Iterable[ModelIdentity | str] | None,
    location: str,
) -> frozenset[ModelIdentity] | None:
    """Normalize one optional exact identity-set constraint."""
    if values is None:
        return None
    identities = []
    for index, value in enumerate(values):
        identities.append(_parse_identity(value, f"{location}[{index}]"))
    if len(set(identities)) != len(identities):
        raise PackageAssemblyError(f"{location} contains duplicate identities")
    return frozenset(identities)


def _require_identity_set(
    actual: Iterable[ModelIdentity],
    expected: frozenset[ModelIdentity] | None,
    label: str,
) -> None:
    """Require an exact identity set and report missing and unexpected models."""
    if expected is None:
        return
    actual_set = frozenset(actual)
    if actual_set == expected:
        return
    missing = sorted((identity.artifact_key for identity in expected - actual_set))
    unexpected = sorted((identity.artifact_key for identity in actual_set - expected))
    raise PackageAssemblyError(
        f"{label} identities do not match: missing={missing}, unexpected={unexpected}"
    )


def _child_identity(path: Path) -> tuple[ModelIdentity, str, bytes]:
    """Read one child archive and return its identity, version, and payload."""
    try:
        members = read_model_archive(path)
        _, config = load_model_config_bytes(
            members["flagtune_config.yaml"], source=str(path)
        )
        identity = model_identity_from_config(config)
        version = config.get("model_version")
    except (KeyError, ModelArchiveError, TypeError, ValueError) as exc:
        raise PackageAssemblyError(
            f"invalid child model archive {path}: {exc}"
        ) from exc
    if not isinstance(version, str) or not version:
        raise PackageAssemblyError(f"child model archive has no model_version: {path}")
    return identity, version, path.read_bytes()


def _base_children(
    package_path: Path,
    *,
    platform_key: str,
    version: str,
) -> dict[ModelIdentity, bytes]:
    """Read all indexed children from an existing platform package."""
    try:
        package = read_platform_package(
            package_path,
            expected_platform_key=platform_key,
            expected_version=version,
        )
    except (OSError, ModelArchiveError, ValueError) as exc:
        raise PackageAssemblyError(
            f"invalid base platform package {package_path}: {exc}"
        ) from exc

    children: dict[ModelIdentity, bytes] = {}
    for artifact_key, payload in package.archives.items():
        try:
            members = read_model_archive_bytes(
                payload, source=f"{package_path}:{artifact_key}"
            )
            _, config = load_model_config_bytes(
                members["flagtune_config.yaml"],
                source=f"{package_path}:{artifact_key}",
            )
            identity = model_identity_from_config(config)
        except (KeyError, ModelArchiveError, TypeError, ValueError) as exc:
            raise PackageAssemblyError(
                f"invalid child {artifact_key!r} in base package {package_path}: {exc}"
            ) from exc
        if identity in children:
            raise PackageAssemblyError(
                f"base package contains duplicate model identity: {identity.artifact_key}"
            )
        children[identity] = payload
    return children


def assemble_platform_package(
    model_paths: Iterable[Path | str],
    *,
    output_path: Path | str,
    base_package: Path | str | None = None,
    expected_model_identities: Iterable[ModelIdentity | str] | None = None,
    required_package_identities: Iterable[ModelIdentity | str] | None = None,
) -> Path:
    """Combine child model archives into one deterministic platform package.

    Args:
        model_paths: Newly trained child ``model.tar.gz`` paths.  All children
            must declare the same platform and model version.
        output_path: Canonical outer package path, for example
            ``nvidia-h20_v1.0.0.tar.gz``.
        base_package: Optional existing outer package.  Its children are kept,
            and a newly supplied child replaces one with the same identity.
        expected_model_identities: Optional exact identity set required from the
            newly supplied child archives.
        required_package_identities: Optional exact identity set required after
            merging the base package and newly supplied children.

    Returns:
        The atomically written platform package path.

    Raises:
        PackageAssemblyError: If paths, identities, versions, or output naming
            are inconsistent.
    """
    paths = [Path(path).expanduser().resolve() for path in model_paths]
    if not paths:
        raise PackageAssemblyError("at least one child model archive is required")
    expected_models = _identity_set(
        expected_model_identities, "expected_model_identities"
    )
    required_package = _identity_set(
        required_package_identities, "required_package_identities"
    )
    inspected = [_child_identity(path) for path in paths]
    platform_key = inspected[0][0].platform_key
    version = inspected[0][1]
    children: dict[ModelIdentity, bytes] = {}
    for path, (identity, child_version, payload) in zip(paths, inspected):
        if identity.platform_key != platform_key:
            raise PackageAssemblyError(
                f"child platform mismatch: {identity.platform_key!r} != {platform_key!r} ({path})"
            )
        if child_version != version:
            raise PackageAssemblyError(
                f"child model version mismatch: {child_version!r} != {version!r} ({path})"
            )
        if identity in children:
            raise PackageAssemblyError(
                f"duplicate child model identity: {identity.artifact_key}"
            )
        children[identity] = payload
    _require_identity_set(children, expected_models, "new model")

    if base_package is not None:
        base_path = Path(base_package).expanduser().resolve()
        base_children = _base_children(
            base_path, platform_key=platform_key, version=version
        )
        children = {**base_children, **children}
    _require_identity_set(children, required_package, "platform package")

    output = Path(output_path).expanduser().resolve()
    expected_name = platform_package_name(platform_key, version)
    if output.name != expected_name:
        raise PackageAssemblyError(
            f"output filename must be {expected_name!r}, got {output.name!r}"
        )

    with tempfile.TemporaryDirectory(prefix="flagtune-package-") as temporary_dir:
        temporary = Path(temporary_dir)
        archive_paths: dict[ModelIdentity, Path] = {}
        ordered_children = sorted(
            children.items(), key=lambda item: item[0].artifact_key
        )
        for index, (identity, payload) in enumerate(ordered_children):
            child_path = temporary / f"child-{index}.tar.gz"
            child_path.write_bytes(payload)
            archive_paths[identity] = child_path
        try:
            return write_platform_package(
                output,
                platform_key=platform_key,
                package_version=version,
                model_archives=archive_paths,
                required_identities=tuple(archive_paths),
            )
        except (ModelArchiveError, OSError, TypeError, ValueError) as exc:
            raise PackageAssemblyError(
                f"cannot write platform package {output}: {exc}"
            ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble one platform package from repeated ``--model`` arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble FlagTune child model archives into one platform package"
    )
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-package", type=Path)
    parser.add_argument(
        "--expected-model-identity",
        action="append",
        help="exact identity required from the newly supplied --model archives",
    )
    parser.add_argument(
        "--required-package-identity",
        action="append",
        help="exact identity required in the assembled platform package",
    )
    args = parser.parse_args(argv)
    try:
        output = assemble_platform_package(
            args.model,
            output_path=args.output,
            base_package=args.base_package,
            expected_model_identities=args.expected_model_identity,
            required_package_identities=args.required_package_identity,
        )
    except (OSError, PackageAssemblyError) as exc:
        print(f"error: {exc}")
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
