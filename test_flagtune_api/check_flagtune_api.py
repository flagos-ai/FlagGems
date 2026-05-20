import importlib
import os
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _clear_flagtune_env():
    os.environ.pop("USE_FLAGTUNE", None)
    os.environ.pop("FLAGTUNE_INCLUDE", None)


def _maybe_force_hopper_arch():
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major >= 9:
                os.environ.setdefault("ARCH", "sm_90")
    except Exception:
        pass


def _apply_flagtune_state(*entries):
    for entry in entries:
        entry._maybe_apply_flagtune()


def main():
    _clear_flagtune_env()
    _maybe_force_hopper_arch()

    import flag_gems

    bmm_mod = importlib.import_module("flag_gems.ops.bmm")
    addmm_mod = importlib.import_module("flag_gems.ops.addmm")
    hopper_mm_mod = importlib.import_module(
        "flag_gems.runtime.backend._nvidia.hopper.ops.mm"
    )

    assert hasattr(flag_gems, "flagtune")
    assert hasattr(flag_gems, "register_flagtune_op")
    assert flag_gems.get_supported_flagtune_ops() == frozenset(("mm", "bmm"))
    assert flag_gems.get_default_flagtune_include() == frozenset(("mm", "bmm"))
    assert set(flag_gems.get_flagtune_registry()) == {"mm", "bmm"}
    assert not flag_gems.runtime.flagtune_enabled("mm")
    assert not flag_gems.runtime.flagtune_enabled("bmm")

    registry_op = "dummy_registry_op"
    spec = flag_gems.register_flagtune_op(
        registry_op,
        default=False,
        description="test-only registry entry",
    )
    assert spec.name == registry_op
    assert registry_op in flag_gems.get_supported_flagtune_ops()
    assert registry_op not in flag_gems.get_default_flagtune_include()
    flag_gems.flagtune(include=registry_op)
    assert flag_gems.runtime.flagtune_enabled(registry_op)
    assert os.environ["FLAGTUNE_INCLUDE"] == registry_op
    try:
        flag_gems.register_flagtune_op(registry_op, default=True)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting registry re-registration should fail")

    bmm_tuner = bmm_mod.bmm_kernel.fn
    mm_tuner = hopper_mm_mod.mm_kernel_general_host_tma.fn
    addmm_tuner = addmm_mod.addmm_kernel.fn

    bmm_default_count = len(bmm_tuner.configs)
    mm_default_count = len(mm_tuner.configs)
    assert bmm_default_count > 0
    assert mm_default_count > 0
    assert getattr(addmm_tuner, "_flagtune_op_name", None) is None

    bmm_expand_path = flag_gems.runtime.config_loader.expand_config_registry["bmm"][
        "expand_yaml_path"
    ]
    expected_hopper_path = (
        _repo_root()
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_nvidia"
        / "hopper"
        / "general_ops_hopper_configs.yaml"
    )
    assert Path(bmm_expand_path).resolve() == expected_hopper_path.resolve()

    flag_gems.flagtune(include="bmm")
    _apply_flagtune_state(bmm_mod.bmm_kernel, hopper_mm_mod.mm_kernel_general_host_tma)
    assert flag_gems.runtime.flagtune_enabled("bmm")
    assert not flag_gems.runtime.flagtune_enabled("mm")
    assert bmm_tuner._flagtune_active
    assert len(bmm_tuner.configs) > bmm_default_count
    bmm_expanded_count = len(bmm_tuner.configs)
    assert not mm_tuner._flagtune_active
    assert len(mm_tuner.configs) == mm_default_count

    flag_gems.flagtune(include="mm")
    _apply_flagtune_state(bmm_mod.bmm_kernel, hopper_mm_mod.mm_kernel_general_host_tma)
    assert not flag_gems.runtime.flagtune_enabled("bmm")
    assert flag_gems.runtime.flagtune_enabled("mm")
    assert not bmm_tuner._flagtune_active
    assert len(bmm_tuner.configs) == bmm_default_count
    assert mm_tuner._flagtune_active
    assert len(mm_tuner.configs) > mm_default_count
    mm_expanded_count = len(mm_tuner.configs)

    try:
        flag_gems.flagtune(include="addmm")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported op addmm should be rejected")

    print("PASS: flagtune API include scope and config switching look correct")
    print(f"bmm configs: {bmm_default_count} -> {bmm_expanded_count}")
    print(f"mm configs: {mm_default_count} -> {mm_expanded_count}")
    print(f"bmm expand yaml: {bmm_expand_path}")


if __name__ == "__main__":
    main()
