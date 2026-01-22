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
