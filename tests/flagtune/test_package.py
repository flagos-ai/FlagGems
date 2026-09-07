"""Test generic assembly of child model archives into platform packages."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

try:
    HAS_FLAGTREE_ARCHIVE = (
        importlib.util.find_spec("triton.flagtune.contract.archive") is not None
    )
except ModuleNotFoundError:
    HAS_FLAGTREE_ARCHIVE = False

pytestmark = pytest.mark.skipif(
    not HAS_FLAGTREE_ARCHIVE,
    reason="platform-package assembly requires the optional FlagTree package",
)

if HAS_FLAGTREE_ARCHIVE:
    from triton.flagtune.contract.archive import (  # noqa: E402
        platform_package_name,
        read_platform_package,
        write_model_archive,
        write_platform_package,
    )
    from triton.flagtune.contract.identity import ModelIdentity  # noqa: E402

    from flag_gems.flagtune.cli.package import (  # noqa: E402
        PackageAssemblyError,
        assemble_platform_package,
    )

PLATFORM = "nvidia-h20"
VERSION = "1.0.0"
GPU = {
    "backend": "cuda",
    "vendor": "NVIDIA",
    "device_name": "NVIDIA H20-3e",
    "architecture": "sm90",
    "platform_key": PLATFORM,
}


def _child(tmp_path: Path, variant: str) -> tuple[ModelIdentity, Path]:
    """Create a minimally valid child archive for package assembly tests."""
    identity = ModelIdentity(
        PLATFORM,
        "flaggems/mul",
        variant,
        "bf16-bf16",
    )
    config = {
        "format_version": 5,
        "model_version": VERSION,
        "platform_key": PLATFORM,
        "op_id": identity.op_id,
        "variant": identity.variant,
        "dtype_key": identity.dtype_key,
        "dtypes": ["bfloat16", "bfloat16"],
        "gpu": GPU,
        "inputs": {"n_elements": {"min": 1}},
        "params": {"BLOCK_SIZE": {"values": [128]}},
        "features": ["n_elements"],
    }
    path = tmp_path / f"{variant}.tar.gz"
    write_model_archive(
        path,
        {
            "xgboost_ranker.json": b"fixture",
            "flagtune_config.yaml": yaml.safe_dump(config).encode("utf-8"),
            "training_summary.json": json.dumps({}).encode("utf-8"),
        },
    )
    return identity, path


def test_assemble_platform_package_preserves_base_and_overlays_children(tmp_path):
    """Keep existing models while replacing or adding newly trained variants."""
    base_identity, base_child = _child(tmp_path, "base")
    new_identity, new_child = _child(tmp_path, "scalar")
    base_path = tmp_path / platform_package_name(PLATFORM, VERSION)
    write_platform_package(
        base_path,
        platform_key=PLATFORM,
        package_version=VERSION,
        model_archives={base_identity: base_child},
        required_identities=(base_identity,),
    )

    output = tmp_path / "assembled" / platform_package_name(PLATFORM, VERSION)
    result = assemble_platform_package(
        [new_child],
        output_path=output,
        base_package=base_path,
        expected_model_identities=(new_identity.artifact_key,),
        required_package_identities=(base_identity, new_identity),
    )
    package = read_platform_package(
        result, expected_platform_key=PLATFORM, expected_version=VERSION
    )
    assert set(package.models) == {
        base_identity.artifact_key,
        new_identity.artifact_key,
    }


def test_assemble_platform_package_rejects_duplicate_children(tmp_path):
    """Reject duplicate identity inputs instead of silently choosing one."""
    _, child = _child(tmp_path, "scalar")
    output = tmp_path / platform_package_name(PLATFORM, VERSION)
    with pytest.raises(PackageAssemblyError, match="duplicate child"):
        assemble_platform_package([child, child], output_path=output)


def test_assemble_platform_package_rejects_unexpected_new_identity(tmp_path):
    """Reject a valid child archive when it is not the requested model."""
    _, child = _child(tmp_path, "scalar")
    expected = ModelIdentity(PLATFORM, "flaggems/mul", "broadcast_2d", "bf16-bf16")
    output = tmp_path / platform_package_name(PLATFORM, VERSION)
    with pytest.raises(PackageAssemblyError, match="new model identities do not match"):
        assemble_platform_package(
            [child],
            output_path=output,
            expected_model_identities=(expected,),
        )


def test_assemble_platform_package_rejects_unexpected_final_identity(tmp_path):
    """Reject missing or extra identities after applying a base package."""
    base_identity, base_child = _child(tmp_path, "base")
    new_identity, new_child = _child(tmp_path, "scalar")
    base_path = tmp_path / platform_package_name(PLATFORM, VERSION)
    write_platform_package(
        base_path,
        platform_key=PLATFORM,
        package_version=VERSION,
        model_archives={base_identity: base_child},
        required_identities=(base_identity,),
    )
    output = tmp_path / "assembled" / platform_package_name(PLATFORM, VERSION)
    with pytest.raises(
        PackageAssemblyError, match="platform package identities do not match"
    ):
        assemble_platform_package(
            [new_child],
            output_path=output,
            base_package=base_path,
            required_package_identities=(new_identity,),
        )
