import pytest

from flag_gems.runtime import backend


def test_non_nvidia_heuristic_import_failure_does_not_fallback(monkeypatch):
    imported_modules = []
    import_module = backend.importlib.import_module

    def fake_import_module(module_name):
        imported_modules.append(module_name)
        if module_name == "_mthreads.heuristics_config_utils":
            raise ModuleNotFoundError("simulated mthreads heuristic import failure")
        return import_module(module_name)

    monkeypatch.setattr(backend.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="mthreads heuristic"):
        backend.get_heuristic_config("mthreads")

    assert "_nvidia.heuristics_config_utils" not in imported_modules


def test_mthreads_softmax_non_inner_uses_mthreads_heuristic():
    configs = backend.get_heuristic_config("mthreads")
    tile_k = configs["softmax_non_inner"]["TILE_K"]

    assert tile_k.__module__ == "_mthreads.heuristics_config_utils"
