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

    default_ops = frozenset(("mm", "bmm"))
    supported_ops = default_ops | {"w8a8_block_fp8_matmul"}

    bmm_mod = importlib.import_module("flag_gems.ops.bmm")
    addmm_mod = importlib.import_module("flag_gems.ops.addmm")
    hopper_mm_mod = importlib.import_module(
        "flag_gems.runtime.backend._nvidia.hopper.ops.mm"
    )
    hopper_w8a8_mod = importlib.import_module(
        "flag_gems.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul"
    )

    assert hasattr(flag_gems, "flagtune")
    assert hasattr(flag_gems, "register_flagtune_op")
    assert flag_gems.get_supported_flagtune_ops() == supported_ops
    assert flag_gems.get_default_flagtune_include() == default_ops
    assert set(flag_gems.get_flagtune_registry()) == set(supported_ops)
    assert not flag_gems.runtime.flagtune_enabled("mm")
    assert not flag_gems.runtime.flagtune_enabled("bmm")
    assert not flag_gems.runtime.flagtune_enabled("w8a8_block_fp8_matmul")

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
    w8a8_general_tuner = hopper_w8a8_mod.w8a8_block_fp8_matmul_kernel_general.fn
    w8a8_splitk_tuner = hopper_w8a8_mod.w8a8_block_fp8_matmul_kernel_splitk.fn
    addmm_tuner = addmm_mod.addmm_kernel.fn

    bmm_default_count = len(bmm_tuner.configs)
    mm_default_count = len(mm_tuner.configs)
    w8a8_general_default_count = len(w8a8_general_tuner.configs)
    w8a8_splitk_default_count = len(w8a8_splitk_tuner.configs)
    assert bmm_default_count > 0
    assert mm_default_count > 0
    assert w8a8_general_default_count > 0
    assert w8a8_splitk_default_count > 0
    assert w8a8_general_tuner._flagtune_op_name == "w8a8_block_fp8_matmul"
    assert w8a8_splitk_tuner._flagtune_op_name == "w8a8_block_fp8_matmul"
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

    flag_gems.flagtune(include="w8a8_block_fp8_matmul")
    _apply_flagtune_state(
        bmm_mod.bmm_kernel,
        hopper_mm_mod.mm_kernel_general_host_tma,
        hopper_w8a8_mod.w8a8_block_fp8_matmul_kernel_general,
        hopper_w8a8_mod.w8a8_block_fp8_matmul_kernel_splitk,
    )
    assert flag_gems.runtime.flagtune_enabled("w8a8_block_fp8_matmul")
    assert not flag_gems.runtime.flagtune_enabled("mm")
    assert not flag_gems.runtime.flagtune_enabled("bmm")
    assert not bmm_tuner._flagtune_active
    assert not mm_tuner._flagtune_active
    assert w8a8_general_tuner._flagtune_active
    assert w8a8_splitk_tuner._flagtune_active
    assert len(w8a8_general_tuner.configs) > w8a8_general_default_count
    assert len(w8a8_splitk_tuner.configs) > w8a8_splitk_default_count
    w8a8_general_expanded_count = len(w8a8_general_tuner.configs)
    w8a8_splitk_expanded_count = len(w8a8_splitk_tuner.configs)

    try:
        flag_gems.flagtune(include="addmm")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported op addmm should be rejected")

    print("PASS: flagtune API include scope and config switching look correct")
    print(f"bmm configs: {bmm_default_count} -> {bmm_expanded_count}")
    print(f"mm configs: {mm_default_count} -> {mm_expanded_count}")
    print(
        "w8a8 configs: "
        f"general {w8a8_general_default_count} -> {w8a8_general_expanded_count}, "
        f"splitk {w8a8_splitk_default_count} -> {w8a8_splitk_expanded_count}"
    )
    print(f"bmm expand yaml: {bmm_expand_path}")


if __name__ == "__main__":
    main()
