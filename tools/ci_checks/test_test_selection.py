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

"""Host-only CI selection checks; no torch, kernels or device are imported."""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RESOLVER = REPO / "tools/ci_checks/derive_changed_operators.py"
spec = importlib.util.spec_from_file_location("ci_selection", RESOLVER)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)

FAKE_PYTEST = r"""#!/bin/bash
[[ "$1" == erase ]] && exit 0
printf '%s\n' "$*" >> calls.log
report=''
for arg in "$@"; do
  case "$arg" in --junitxml=*) report=${arg#--junitxml=} ;; esac
done
[[ -n "$report" ]] || exit 12
case "$STUB_MODE" in
  missing) exit 0 ;;
  empty|empty_zero) xml='<testsuites><testsuite tests="0"/></testsuites>' ;;
  skipped) xml='<testsuites><testsuite tests="1"><testcase><skipped/></testcase></testsuite></testsuites>' ;;
  failure_zero) xml='<testsuites><testsuite tests="1"><testcase><failure/></testcase></testsuite></testsuites>' ;;
  malformed) xml='not XML' ;;
  mixed) xml='<testsuites><testsuite tests="2"><testcase/><testcase><skipped/></testcase></testsuite></testsuites>' ;;
  *) xml='<testsuites><testsuite tests="1"><testcase/></testsuite></testsuites>' ;;
esac
printf '%s\n' "$xml" > "$report"
[[ "$STUB_MODE" == empty ]] && exit 5
[[ "$STUB_MODE" == nonzero_pass ]] && exit 1
exit 0
"""


def bash_executable():
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parent.parent / "bin/bash.exe"
            if candidate.is_file():
                return str(candidate)
    bash = shutil.which("bash")
    assert bash, "The CI shell integration checks require bash"
    return bash


@pytest.fixture
def project(tmp_path):
    (tmp_path / "tools/ci_checks").mkdir(parents=True)
    (tmp_path / "conf").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copyfile(RESOLVER, tmp_path / "tools/ci_checks/derive_changed_operators.py")
    shutil.copyfile(REPO / "tools/test-op.sh", tmp_path / "tools/test-op.sh")
    (tmp_path / "conf/ci_test_aliases.yaml").write_text(
        "addmm_out: test_addmm\nupsample_linear1d_backward: test_upsample_linear1d\n",
        encoding="utf-8",
    )
    names = [
        "test_linear.py",
        "test_quant.py",
        "test_div.py",
        "test_divide.py",
        "test_true_divide.py",
        "test_trunc_divide.py",
        "test_floor_divide.py",
        "test_remainder.py",
        "test_addmm.py",
        "test_mm.py",
        "test_rnn_relu.py",
        "test_flash_mla_sparse_fwd.py",
        "test_scaled_dot_product_attention.py",
        "test_flash_attention_backward.py",
        "test_flash_attention.py",
        "test_flash_attn_varlen_func.py",
        "test_scaled_dot_product_cudnn_attention.py",
        "test_scaled_dot_product_fused_attention_overrideable.py",
        "test_upsample_linear1d.py",
    ]
    for name in names:
        (tmp_path / "tests" / name).write_text("", encoding="utf-8")
    (tmp_path / "tests/test_DSA").mkdir()
    (tmp_path / "tests/test_DSA/test_indexer_k_tiled.py").write_text(
        "", encoding="utf-8"
    )
    for name in ("coverage", "pytest"):
        target = tmp_path / "bin" / name
        target.write_text(FAKE_PYTEST, encoding="utf-8", newline="\n")
        target.chmod(0o755)
    python_shim = tmp_path / "bin/python3"
    python_shim.write_text(
        f'#!/bin/bash\nexec {shlex.quote(Path(sys.executable).as_posix())} "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    python_shim.chmod(0o755)
    return tmp_path


def run_ci(project, changed, mode="pass", *, refs=(), metadata=None):
    env = os.environ.copy()
    env.update(CHANGED_FILES=" ".join(changed), GITHUB_SHA="123456789", STUB_MODE=mode)
    if metadata:
        env.update(metadata)
    result = subprocess.run(
        [
            bash_executable(),
            "--noprofile",
            "--norc",
            "-c",
            'export PATH="$PWD/bin:$PATH"; bash tools/test-op.sh 123 '
            + shlex.join(refs),
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    call_file = project / "calls.log"
    calls = (
        call_file.read_text(encoding="utf-8").splitlines() if call_file.exists() else []
    )
    return result, calls


@pytest.mark.parametrize(
    "sources,expected",
    [
        (
            ["src/flag_gems/runtime/backend/_kunlunxin/ops/linear.py"],
            ["test_linear.py"],
        ),
        (
            ["src/flag_gems/runtime/backend/_kunlunxin/fused/flashmla_sparse.py"],
            ["test_flash_mla_sparse_fwd.py"],
        ),
        (
            ["src/flag_gems/runtime/backend/_kunlunxin/ops/attention.py"],
            [
                "test_scaled_dot_product_attention.py",
                "test_flash_attention_backward.py",
                "test_flash_attention.py",
                "test_flash_attn_varlen_func.py",
            ],
        ),
        (
            [
                "src/flag_gems/runtime/backend/_kunlunxin/ops/flash_attention_backward.py"
            ],
            ["test_flash_attention_backward.py"],
        ),
        (
            [
                "src/flag_gems/runtime/backend/_kunlunxin/ops/_scaled_dot_product_cudnn_attention.py",
                "src/flag_gems/runtime/backend/_kunlunxin/ops/_scaled_dot_product_fused_attention_overrideable.py",
            ],
            [
                "test_scaled_dot_product_cudnn_attention.py",
                "test_scaled_dot_product_fused_attention_overrideable.py",
            ],
        ),
        (
            [
                "src/flag_gems/ops/rnn_relu.py",
                "src/flag_gems/runtime/backend/_kunlunxin/ops/addmm.py",
                "src/flag_gems/runtime/backend/_kunlunxin/ops/mm.py",
            ],
            ["test_rnn_relu.py", "test_addmm.py", "test_mm.py"],
        ),
        (
            ["src/flag_gems/runtime/backend/_kunlunxin/fused/DSA/indexer_k_tiled.py"],
            ["test_DSA/test_indexer_k_tiled.py"],
        ),
        (["src/flag_gems/ops/addmm_out.py"], ["test_addmm.py"]),
        (["src/flag_gems/fused/linear.py"], ["test_linear.py"]),
        (
            ["src/flag_gems/runtime/backend/_kunlunxin/ops/div.py"],
            [
                "test_div.py",
                "test_divide.py",
                "test_true_divide.py",
                "test_trunc_divide.py",
                "test_floor_divide.py",
                "test_remainder.py",
            ],
        ),
        (
            ["src/flag_gems/ops/upsample_linear1d_backward.py"],
            ["test_upsample_linear1d.py"],
        ),
    ],
)
def test_source_only_changes_run_full_correctness_files(project, sources, expected):
    result, calls = run_ci(project, sources)
    assert result.returncode == 0, result.stdout + result.stderr
    full_calls = [call for call in calls if "--quick" not in call]
    assert len(full_calls) == len(expected)
    for name in expected:
        assert any(f"tests/{name} " in call for call in full_calls)
    assert all("--timeout=900" in call and "--junitxml=" in call for call in calls)
    assert all(
        " -m " not in call.replace("-m pytest", "") and " -k " not in call
        for call in calls
    )


def test_explicit_and_derived_file_run_once(project):
    result, calls = run_ci(
        project, ["tests/test_addmm.py", "src/flag_gems/ops/addmm.py"]
    )
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2  # One complete run and the existing quick CPU run.
    assert sum("--quick" in call for call in calls) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "empty",
        "empty_zero",
        "skipped",
        "missing",
        "failure_zero",
        "malformed",
        "nonzero_pass",
    ],
)
def test_unexecuted_or_failed_results_cannot_pass(project, mode):
    result, calls = run_ci(project, ["src/flag_gems/ops/linear.py"], mode)
    assert result.returncode != 0
    assert len(calls) == 1


def test_preexisting_partial_skips_are_allowed(project):
    result, calls = run_ci(project, ["src/flag_gems/ops/linear.py"], "mixed")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2


@pytest.mark.parametrize(
    "changes",
    [
        ["src/flag_gems/runtime/backend/_kunlunxin/fused/unknown.py"],
        ["src/flag_gems/runtime/backend/_kunlunxin/ops/__init__.py"],
        ["src/flag_gems/runtime/backend/_kunlunxin/monkey_patch.py"],
        ["src/flag_gems/__init__.py"],
        ["src/flag_gems/utils/helper.py"],
        ["src/flag_gems/runtime/backend/_kunlunxin/tune_configs.yaml"],
        [
            "src/flag_gems/runtime/backend/_kunlunxin/monkey_patch.py",
            "tests/test_quant.py",
        ],
        [
            "src/flag_gems/runtime/backend/_kunlunxin/monkey_patch.py",
            "src/flag_gems/runtime/backend/_kunlunxin/ops/linear.py",
        ],
        [
            "src/flag_gems/runtime/backend/_kunlunxin/monkey_patch.py",
            "tests/test_linear.py",
        ],
        [
            "src/flag_gems/runtime/backend/_kunlunxin/tune_configs.yaml",
            "src/flag_gems/runtime/backend/_kunlunxin/ops/linear.py",
        ],
        [
            "src/flag_gems/runtime/backend/_kunlunxin/ops/__init__.py",
            "src/flag_gems/runtime/backend/_kunlunxin/ops/linear.py",
            "tests/test_linear.py",
        ],
        ["src/flag_gems/ops/unknown.py", "tests/test_linear.py"],
        ["src/flag_gems/ops/unknown.py", "benchmark/test_linear.py"],
    ],
)
def test_missing_source_coverage_cannot_pass(project, changes):
    result, calls = run_ci(project, changes)
    assert result.returncode != 0
    assert not calls


def test_alias_parse_error_is_not_swallowed(project):
    (project / "conf/ci_test_aliases.yaml").write_text("broken: [", encoding="utf-8")
    result, calls = run_ci(project, ["src/flag_gems/ops/linear.py"])
    assert result.returncode != 0
    assert not calls


def test_missing_mapped_file_is_not_ignored(project):
    (project / "tests/test_flash_attention_backward.py").unlink()
    result, calls = run_ci(
        project, ["src/flag_gems/runtime/backend/_kunlunxin/ops/attention.py"]
    )
    assert result.returncode != 0
    assert not calls


def test_documentation_only_change_needs_no_tests(project):
    result, calls = run_ci(
        project, ["README.md", "src/flag_gems/runtime/backend/_kunlunxin/README.md"]
    )
    assert result.returncode == 0, result.stderr
    assert not calls


def test_explicit_quant_test_is_not_discarded(project):
    result, calls = run_ci(project, ["tests/test_quant.py"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert "tests/test_quant.py " in calls[0]


def test_missing_div_family_file_cannot_be_replaced_by_div_tests(project):
    (project / "tests/test_remainder.py").unlink()
    result, calls = run_ci(
        project, ["src/flag_gems/runtime/backend/_kunlunxin/ops/div.py"]
    )
    assert result.returncode != 0
    assert not calls


def test_deleted_correctness_test_fails_before_execution(project):
    (project / "tests/test_linear.py").unlink()
    result, calls = run_ci(project, ["tests/test_linear.py"])
    assert result.returncode != 0
    assert not calls


def test_deleted_source_still_requires_its_complete_test_file(project):
    source = project / "src/flag_gems/ops/linear.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    source.unlink()
    result, calls = run_ci(project, ["src/flag_gems/ops/linear.py"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert all("tests/test_linear.py " in call for call in calls)


def test_test_selection_diff_includes_deletions(monkeypatch):
    commands = []

    def diff_result(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, "src/flag_gems/ops/deleted_op.py\n", ""
        )

    monkeypatch.setattr(selection.subprocess, "run", diff_result)
    assert selection.get_diff_files("base", "head", include_deleted=True) == [
        "src/flag_gems/ops/deleted_op.py"
    ]
    assert "--diff-filter=ACMRD" in commands[0]
    selection.get_diff_files("base", "head")
    assert "--diff-filter=ACMR" in commands[1]


def test_registry_derivation_retains_existing_rule_check_scope():
    paths = [
        "src/flag_gems/fused/foo.py",
        "src/flag_gems/runtime/backend/_kunlunxin/fused/DSA/bar.py",
        "src/flag_gems/runtime/backend/_nvidia/hopper/ops/mm.py",
        "src/flag_gems/runtime/backend/_kunlunxin/ops/__init__.py",
    ]
    assert selection.derive_operators(paths, {}) == ["mm"]
    assert (
        selection.expected_marker("_stack", {"stack", "_stack"}) == "underscore_stack"
    )


def test_failed_git_diff_is_not_an_empty_change_list(monkeypatch):
    def failed_diff(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0], stderr="unknown revision")

    monkeypatch.setattr(selection.subprocess, "run", failed_diff)
    with pytest.raises(SystemExit) as exc:
        selection.get_diff_files("missing", "HEAD")
    assert exc.value.code == 2


@pytest.mark.parametrize("tag", ["error", "failure"])
def test_suite_level_failure_cannot_be_hidden_by_passed_case(tmp_path, tag):
    report = tmp_path / "results.xml"
    report.write_text(
        f"<testsuites><testsuite><testcase/><{tag}/></testsuite></testsuites>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Failed tests"):
        selection.validate_junit(str(report))


OPS_INIT = "src/flag_gems/runtime/backend/_kunlunxin/ops/__init__.py"
OPS = str(Path(OPS_INIT).parent.as_posix())


def git(project, *args):
    result = subprocess.run(
        ["git", *args], cwd=project, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def write_source(project, path, content):
    target = project / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def commit_fixture(project, message):
    # Only disposable pytest fixture repositories are committed.
    git(project, "add", ".")
    git(project, "commit", "-q", "-m", message)
    return git(project, "rev-parse", "HEAD")


@pytest.fixture
def history(project, monkeypatch):
    git(project, "init", "-q")
    git(project, "config", "user.name", "CI fixture")
    git(project, "config", "user.email", "fixture@example.invalid")
    git(project, "config", "commit.gpgsign", "false")
    git(project, "config", "core.autocrlf", "false")
    for name in ("linear", "mm", "div"):
        write_source(project, f"{OPS}/{name}.py", f"def {name}(): pass\n")
    write_source(
        project, OPS_INIT, "from .linear import linear\n__all__ = ['linear']\n"
    )
    base = commit_fixture(project, "base")
    monkeypatch.chdir(project)
    for name in (
        "FLAGGEMS_CI_EVENT",
        "FLAGGEMS_CI_BASE_SHA",
        "FLAGGEMS_CI_PR_HEAD_SHA",
        "FLAGGEMS_CI_TESTED_SHA",
    ):
        monkeypatch.delenv(name, raising=False)
    return project, base


def test_import_only_addition_runs_complete_file_with_explicit_refs(history):
    project, base = history
    write_source(
        project,
        OPS_INIT,
        "from .linear import linear\nfrom .mm import mm\n__all__ = ['linear']\n",
    )
    head = commit_fixture(project, "add import")
    result, calls = run_ci(project, [OPS_INIT], refs=("--base", base, "--head", head))
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 2
    assert all("tests/test_mm.py " in call for call in calls)


def test_removed_export_requires_full_div_family(history):
    project, _ = history
    write_source(
        project,
        OPS_INIT,
        "from .div import div, remainder\n__all__ = ['div', 'remainder']\n",
    )
    base = commit_fixture(project, "div exports")
    write_source(
        project, OPS_INIT, "from .div import div, remainder\n__all__ = ['div']\n"
    )
    head = commit_fixture(project, "remove remainder export")
    assert selection.derive_test_files([OPS_INIT], base=base, head=head) == sorted(
        selection.BACKEND_TEST_FILES[f"{OPS}/div.py"]
    )


def test_changed_alias_covers_old_and_new_modules(history):
    project, base = history
    write_source(
        project, OPS_INIT, "from .mm import mm as linear\n__all__ = ['linear']\n"
    )
    head = commit_fixture(project, "replace alias")
    assert selection.derive_test_files([OPS_INIT], base=base, head=head) == [
        "tests/test_linear.py",
        "tests/test_mm.py",
    ]


def test_removed_import_and_deleted_module_still_select_tests(history):
    project, base = history
    write_source(project, OPS_INIT, "__all__ = []\n")
    (project / f"{OPS}/linear.py").unlink()
    head = commit_fixture(project, "remove implementation")
    assert selection.derive_test_files([OPS_INIT], base=base, head=head) == [
        "tests/test_linear.py"
    ]


@pytest.mark.parametrize("remove_all", [False, True])
def test_explicit_implicit_exports_transition_covers_retained_imports(
    history, remove_all
):
    project, _ = history
    implicit = "from .linear import linear\n"
    explicit = implicit + "__all__ = []\n"
    write_source(project, OPS_INIT, explicit if remove_all else implicit)
    base = commit_fixture(project, "initial exports")
    write_source(project, OPS_INIT, implicit if remove_all else explicit)
    head = commit_fixture(project, "change export mode")
    assert selection.derive_test_files([OPS_INIT], base=base, head=head) == [
        "tests/test_linear.py"
    ]


@pytest.mark.parametrize(
    "replacement",
    [
        "from .linear import linear\n__all__ = ['linear']\ninstall_monkey_patch()\n",
        "from .linear import linear\n__all__ = get_exports()\n",
        "from .linear import *\n__all__ = ['linear']\n",
        "from .linear import linear\nfrom .mm import mm as linear\n__all__ = ['linear']\n",
        "from .linear import linear\n__all__ = ['linear', 'linear']\n",
        "from .linear import linear\n__all__ = ['linear', 'unmapped']\n",
        "from .linear import linear\n__all__ = ['linear']\n__all__.append('mm')\n",
    ],
)
def test_unproven_initializer_deltas_fail_closed(history, replacement):
    project, base = history
    write_source(project, OPS_INIT, replacement)
    head = commit_fixture(project, "unproven delta")
    with pytest.raises(ValueError):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


def test_retained_import_cannot_move_across_executable_statement(history):
    project, _ = history
    write_source(
        project,
        OPS_INIT,
        "from .linear import linear\ninstall()\n__all__ = ['linear']\n",
    )
    base = commit_fixture(project, "before call")
    write_source(
        project,
        OPS_INIT,
        "install()\nfrom .linear import linear\n__all__ = ['linear']\n",
    )
    head = commit_fixture(project, "after call")
    with pytest.raises(ValueError, match="ordering changed"):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


def test_retained_export_order_cannot_change(history):
    project, _ = history
    prefix = "from .linear import linear\nfrom .mm import mm\n"
    write_source(project, OPS_INIT, prefix + "__all__ = ['linear', 'mm']\n")
    base = commit_fixture(project, "before order")
    write_source(project, OPS_INIT, prefix + "__all__ = ['mm', 'linear']\n")
    head = commit_fixture(project, "after order")
    with pytest.raises(ValueError, match="exports were reordered"):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


def test_static_package_reexport_selects_defining_module(history):
    project, _ = history
    package = f"{OPS}/bundle/__init__.py"
    write_source(project, package, "from ..mm import mm as matrix\n")
    base = commit_fixture(project, "package")
    write_source(
        project,
        OPS_INIT,
        "from .linear import linear\nfrom .bundle import matrix as product\n__all__ = ['linear', 'product']\n",
    )
    head = commit_fixture(project, "reexport")
    assert selection.derive_test_files([OPS_INIT], base=base, head=head) == [
        "tests/test_mm.py"
    ]
    write_source(project, package, "from ..linear import linear as matrix\n")
    with pytest.raises(ValueError, match="Tested file differs"):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


def test_ambiguous_module_and_package_import_fails_closed(history):
    project, _ = history
    write_source(project, f"{OPS}/bundle.py", "matrix = 1\n")
    write_source(
        project, f"{OPS}/bundle/__init__.py", "from ..mm import mm as matrix\n"
    )
    base = commit_fixture(project, "ambiguous module package")
    write_source(
        project, OPS_INIT, "from .bundle import matrix\n__all__ = ['matrix']\n"
    )
    head = commit_fixture(project, "import ambiguous name")
    with pytest.raises(ValueError, match="Ambiguous module/package"):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


def test_untracked_package_cannot_shadow_proven_module(history):
    project, base = history
    write_source(project, OPS_INIT, "from .mm import mm\n__all__ = ['mm']\n")
    head = commit_fixture(project, "import module")
    write_source(project, f"{OPS}/mm/__init__.py", "mm = replacement\n")
    with pytest.raises(ValueError, match="Tested file differs"):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


@pytest.mark.parametrize(
    "path", [OPS_INIT, f"{OPS}/mm.py", "tests/test_mm.py", "conf/ci_test_aliases.yaml"]
)
def test_dirty_tested_initializer_source_or_test_is_rejected(history, path):
    project, base = history
    write_source(
        project,
        OPS_INIT,
        "from .linear import linear\nfrom .mm import mm\n__all__ = ['linear', 'mm']\n",
    )
    head = commit_fixture(project, "add mm")
    with (project / path).open("a", encoding="utf-8") as stream:
        stream.write("# uncommitted content\n")
    with pytest.raises(ValueError, match="Tested file differs"):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


def test_initializer_without_history_fails_before_pytest(history):
    project, _ = history
    result, calls = run_ci(project, [OPS_INIT])
    assert result.returncode != 0
    assert "requires explicit refs or PR merge metadata" in result.stderr
    assert not calls


def test_explicit_context_uses_merge_base_and_requires_tested_head(history):
    project, base = history
    git(project, "checkout", "-q", "-b", "diverged-base")
    write_source(project, "README.md", "base-only change\n")
    later_base = commit_fixture(project, "base advanced")
    git(project, "checkout", "-q", "-b", "feature", base)
    write_source(
        project,
        OPS_INIT,
        "from .linear import linear\nfrom .mm import mm\n__all__ = ['linear']\n",
    )
    head = commit_fixture(project, "feature")
    assert selection.initializer_context(later_base, head) == (base, head)
    with pytest.raises(ValueError, match="does not match"):
        selection.initializer_context(base, later_base)
    with pytest.raises(ValueError):
        selection.initializer_context("missing-commit", head)
    with pytest.raises(ValueError, match="both --base and --head"):
        selection.initializer_context(base, None)


def merge_fixture(project, base):
    git(project, "checkout", "-q", "-b", "feature")
    write_source(
        project,
        OPS_INIT,
        "from .linear import linear\nfrom .mm import mm\n__all__ = ['linear']\n",
    )
    pr_head = commit_fixture(project, "feature")
    git(project, "checkout", "-q", "-b", "tested-merge", base)
    git(project, "merge", "-q", "--no-ff", "feature", "-m", "tested merge")
    head = git(project, "rev-parse", "HEAD")
    return {
        "FLAGGEMS_CI_EVENT": "pull_request",
        "FLAGGEMS_CI_BASE_SHA": base,
        "FLAGGEMS_CI_PR_HEAD_SHA": pr_head,
        "FLAGGEMS_CI_TESTED_SHA": head,
    }


def test_verified_pr_merge_metadata_runs_tests(history):
    project, base = history
    metadata = merge_fixture(project, base)
    result, calls = run_ci(project, [OPS_INIT], metadata=metadata)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 2
    assert all("tests/test_mm.py " in call for call in calls)


@pytest.mark.parametrize("mismatch", ["base", "head", "tested", "event"])
def test_inconsistent_pr_metadata_is_rejected(history, monkeypatch, mismatch):
    project, base = history
    metadata = merge_fixture(project, base)
    key = {
        "base": "FLAGGEMS_CI_BASE_SHA",
        "head": "FLAGGEMS_CI_PR_HEAD_SHA",
        "tested": "FLAGGEMS_CI_TESTED_SHA",
        "event": "FLAGGEMS_CI_EVENT",
    }[mismatch]
    metadata[key] = "push" if mismatch == "event" else "0" * 40
    for name, value in metadata.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        selection.initializer_context(None, None)


def test_shallow_checkout_needs_both_merge_parents(history, tmp_path, monkeypatch):
    project, base = history
    metadata = merge_fixture(project, base)
    clone = tmp_path / "shallow"
    git(project, "clone", "-q", "--depth=1", project.as_uri(), str(clone))
    monkeypatch.chdir(clone)
    for name, value in metadata.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        selection.initializer_context(None, None)
    git(clone, "fetch", "-q", "--depth=2", "origin")
    assert selection.initializer_context(None, None) == (
        base,
        metadata["FLAGGEMS_CI_TESTED_SHA"],
    )


@pytest.mark.parametrize(
    "content",
    [
        "from ..mm import mm as matrix\nmatrix = replacement\n",
        "from . import matrix\n",
        "from .nested import matrix\n",
    ],
)
def test_dynamic_missing_or_cyclic_reexports_fail_closed(history, content):
    project, _ = history
    write_source(project, f"{OPS}/bundle/__init__.py", content)
    write_source(project, f"{OPS}/bundle/nested/__init__.py", "from .. import matrix\n")
    base = commit_fixture(project, "unproven package")
    write_source(
        project,
        OPS_INIT,
        "from .linear import linear\nfrom .bundle import matrix\n__all__ = ['linear', 'matrix']\n",
    )
    head = commit_fixture(project, "unproven reexport")
    with pytest.raises(ValueError):
        selection.derive_test_files([OPS_INIT], base=base, head=head)


def test_ambiguous_merge_base_is_rejected(history):
    project, base = history
    tree = git(project, "rev-parse", "HEAD^{tree}")
    a1 = git(project, "commit-tree", tree, "-p", base, "-m", "a1")
    b1 = git(project, "commit-tree", tree, "-p", base, "-m", "b1")
    a2 = git(project, "commit-tree", tree, "-p", a1, "-p", b1, "-m", "a2")
    b2 = git(project, "commit-tree", tree, "-p", b1, "-p", a1, "-m", "b2")
    git(project, "checkout", "-q", "--detach", b2)
    assert len(git(project, "merge-base", "--all", a2, b2).splitlines()) == 2
    with pytest.raises(ValueError, match="one distinct merge base"):
        selection.initializer_context(a2, b2)


def test_all_backend_test_jobs_bind_proof_to_the_tested_checkout():
    workflow = selection.yaml.safe_load(
        (REPO / ".github/workflows/backend-test.yaml").read_text(encoding="utf-8")
    )
    checked = []
    for name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if "tools/test-op.sh" not in step.get("run", ""):
                continue
            checked.append(name)
            assert step["env"]["FLAGGEMS_CI_EVENT"] == "${{ github.event_name }}"
            assert (
                step["env"]["FLAGGEMS_CI_BASE_SHA"]
                == "${{ github.event.pull_request.base.sha }}"
            )
            assert (
                step["env"]["FLAGGEMS_CI_PR_HEAD_SHA"]
                == "${{ github.event.pull_request.head.sha }}"
            )
            assert step["env"]["FLAGGEMS_CI_TESTED_SHA"] == "${{ github.sha }}"
            checkouts = [
                item
                for item in job["steps"]
                if "checkout-retry" in item.get("uses", "")
            ]
            assert len(checkouts) == 1
            assert checkouts[0]["with"]["ref"] == "${{ github.sha }}"
            assert checkouts[0]["with"]["fetch-depth"] == 2
    assert checked
