"""Validate FlagGems platform-level Cost Model package detection."""

from types import SimpleNamespace

import pytest

from flag_gems.flagtune.runtime import model_package


@pytest.fixture(autouse=True)
def clear_platform_package_cache(monkeypatch):
    """Isolate process-level package results between tests."""
    monkeypatch.setattr(model_package, "_PLATFORM_PACKAGE_AVAILABILITY", {})


def _manager_raising(error):
    def resolve(*_args, **_kwargs):
        raise error

    return SimpleNamespace(resolve=resolve)


def test_platform_package_probe_uses_outer_package_resolution(monkeypatch):
    """Resolve only the platform package with a valid sentinel identity."""
    calls = []

    def resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return "/cache/nvidia-h20_v1.0.0.tar.gz"

    monkeypatch.setattr(
        model_package,
        "_discover_platform_key",
        lambda: "nvidia-h20",
    )
    monkeypatch.setattr(
        model_package,
        "_get_model_manager",
        lambda: SimpleNamespace(resolve=resolve),
    )

    assert model_package.platform_model_package_available() is True
    assert model_package.platform_model_package_available() is True
    assert calls == [
        (
            ("flaggems/platform-package-probe", "availability"),
            {"platform_key": "nvidia-h20", "dtype_key": "f32"},
        )
    ]


def test_unversioned_manifest_miss_marks_platform_unadapted(monkeypatch):
    """Convert only a missing platform entry into unavailable capability."""
    error = FileNotFoundError(
        "FlagTune Manifest has no package for platform 'metax-c550'; "
        "checked flat user packages and package cache"
    )
    manager = _manager_raising(error)
    monkeypatch.setattr(
        model_package,
        "_discover_platform_key",
        lambda: "metax-c550",
    )
    monkeypatch.setattr(model_package, "_get_model_manager", lambda: manager)

    assert model_package.platform_model_package_available() is False
    assert model_package.platform_model_package_available() is False


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(
            "FlagTune Manifest has no package for platform 'nvidia-h20' "
            "at version '2.0.0'; checked package cache"
        ),
        FileNotFoundError(
            "FlagTune package for platform 'nvidia-h20' is not cached and "
            "FLAGTUNE_DISABLE_REMOTE=1 prevents downloading it"
        ),
        RuntimeError("failed to download FlagTune platform package"),
        RuntimeError("FlagTune platform package SHA-256 mismatch"),
    ],
)
def test_platform_package_probe_propagates_other_failures(monkeypatch, error):
    """Preserve fixed-version, policy, download, and validation failures."""
    monkeypatch.setattr(
        model_package,
        "_discover_platform_key",
        lambda: "nvidia-h20",
    )
    monkeypatch.setattr(
        model_package,
        "_get_model_manager",
        lambda: _manager_raising(error),
    )

    with pytest.raises(type(error)) as exc_info:
        model_package.platform_model_package_available()

    assert exc_info.value is error


def test_model_source_environment_change_rechecks_availability(monkeypatch):
    """Do not reuse a result after selecting a different model cache."""
    calls = 0
    error = FileNotFoundError(
        "FlagTune Manifest has no package for platform 'metax-c550'; checked cache"
    )

    def resolve(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(
        model_package,
        "_discover_platform_key",
        lambda: "metax-c550",
    )
    monkeypatch.setattr(
        model_package,
        "_get_model_manager",
        lambda: SimpleNamespace(resolve=resolve),
    )
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", "/cache/one")

    assert model_package.platform_model_package_available() is False
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", "/cache/two")
    assert model_package.platform_model_package_available() is False
    assert calls == 2
