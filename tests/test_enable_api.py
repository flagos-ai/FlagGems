import re

import pytest
import torch

import flag_gems


def ops_list_to_str(ops_list):
    return "_".join(ops_list).replace(".", "_").replace("-", "_")


def run_ops_and_logs(tmp_path, filename, include=None, exclude=None):
    path_file = tmp_path / filename
    with flag_gems.use_gems(
        include=include, exclude=exclude, record=True, path=path_file
    ):
        a = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
        b = torch.tensor([4.0, 5.0, 6.0], device=flag_gems.device)
        v = torch.tensor(0.5, device=flag_gems.device)
        _ = a + b
        _ = a * b
        _ = torch.sum(a)
        cond = a > 0
        _ = torch.masked_fill(a, ~cond, v)

    assert path_file.exists(), f"Log file {path_file} not found"
    log_content = path_file.read_text()
    return log_content


@pytest.mark.enable
def test_enable(tmp_path):
    log_content = run_ops_and_logs(tmp_path, "gems_enable.log")
    log_prefixes = {
        line.split(":", 1)[0].strip()
        for line in log_content.splitlines()
        if line.strip() and ":" in line
    }
    expected_fragments = [
        "flag_gems.ops.add",
        "flag_gems.ops.mul",
        "flag_gems.ops.sum",
        "flag_gems.ops.gt.gt_scalar",
        "flag_gems.ops.bitwise_not",
        "flag_gems.ops.masked_fill",
    ]
    missing = [
        frag
        for frag in expected_fragments
        if not any(p.startswith(f"[DEBUG] {frag}") for p in log_prefixes)
    ]
    assert not missing, f"Missing expected log entries (prefix match): {missing}"


@pytest.mark.enable_with_exclude
@pytest.mark.parametrize(
    "exclude_op", [["masked_fill", "masked_fill_"], ["mul", "sum", "sum_dim"]]
)
def test_enable_with_exclude(exclude_op, tmp_path):
    log_content = run_ops_and_logs(
        tmp_path,
        f"gems_enable_without_{ops_list_to_str(exclude_op)}.log",
        exclude=exclude_op,
    )

    log_prefixes = {
        line.split(":", 1)[0].strip()
        for line in log_content.splitlines()
        if line.strip() and ":" in line
    }

    for op in exclude_op:
        present = [p for p in log_prefixes if op in p]
        assert not present, f"Found excluded op '{op}' in log file: {present}"


@pytest.mark.only_enable
@pytest.mark.parametrize(
    "include_op", [["sum"], ["mul", "sum"], ["bitwise_not", "masked_fill"]]
)
def test_only_enable(include_op, tmp_path):
    log_content = run_ops_and_logs(
        tmp_path,
        f"gems_only_enable_{ops_list_to_str(include_op)}.log",
        include=include_op,
    )

    pattern = r"flag_gems\.ops\.\w+\.(\w+):"
    found_ops = set(re.findall(pattern, log_content))
    for op in found_ops:
        assert (
            op in include_op
        ), f"Found unexpected op '{op}' in log file. Allowed op: {include_op}"


@pytest.mark.only_enable
def test_only_enable_with_yaml(tmp_path):
    include_ops = ["sum", "mul"]
    yaml_path = tmp_path / "only_enable.yaml"
    yaml_path.write_text("include:\n  - sum\n  - mul\n")

    log_content = run_ops_and_logs(
        tmp_path,
        "gems_only_enable_yaml.log",
        include=str(yaml_path),
    )

    pattern = r"flag_gems\.ops\.\w+\.(\w+):"
    found_ops = set(re.findall(pattern, log_content))

    assert found_ops, "No ops were logged; expected YAML include to be applied"
    assert "sum" in found_ops, "Expected 'sum' to be registered via YAML include"
    assert "mul" in found_ops, "Expected 'mul' to be registered via YAML include"
    unexpected = found_ops - set(include_ops)
    assert not unexpected, f"Found unexpected ops via YAML include: {unexpected}"


@pytest.mark.only_enable
def test_only_enable_default(tmp_path, monkeypatch):
    # Create a mock YAML file with known include ops
    include_ops = ["sum", "mul", "add"]
    yaml_content = "include:\n" + "\n".join(f"  - {op}" for op in include_ops)
    yaml_path = tmp_path / "mock_enable_configs.yaml"
    yaml_path.write_text(yaml_content)

    monkeypatch.setattr(
        flag_gems.config,
        "get_default_enable_config",
        lambda vendor_name, arch_name: [yaml_path],
    )

    log_file = "gems_only_enable_default_mock.log"
    path_file = tmp_path / log_file

    # Map ops to torch functions for dynamic execution
    op_map = {
        "sum": lambda a, b: torch.sum(a),
        "mul": lambda a, b: a * b,
        "add": lambda a, b: a + b,
    }

    with flag_gems.use_gems(include="default", record=True, path=path_file):
        a = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
        b = torch.tensor([4.0, 5.0, 6.0], device=flag_gems.device)
        # Run a couple of ops from the include list to ensure they log.
        for op in include_ops[:2]:
            if op in op_map:
                _ = op_map[op](a, b)
            else:
                # Fallback: exercise a basic op to trigger logging.
                _ = a + b

    assert path_file.exists(), f"Log file {path_file} not found"
    log_content = path_file.read_text()

    pattern = r"flag_gems\.ops\.[^\.]+\.(\w+):"
    found_ops = set(re.findall(pattern, log_content))
    assert found_ops, "No ops were logged for default include"
    assert found_ops.issubset(
        set(include_ops)
    ), f"Found unexpected ops via default include: {found_ops - set(include_ops)}"
