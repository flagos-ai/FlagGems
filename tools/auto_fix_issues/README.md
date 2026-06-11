# auto_fix_issues

Automatically fix FlagGems issues using Claude Code as the coding agent.

## Overview

This tool takes a YAML list of known issues (accuracy failures, runtime errors, etc.) and orchestrates parallel Claude Code sessions — each in its own git worktree — to reproduce, diagnose, and fix the issues.

Each CC session runs with `--dangerously-skip-permissions` (no interactive confirmation) and `--output-format stream-json` for structured logging.

## Quick Start

```bash
# 1. Copy and fill in config
cp config.yaml.example config.yaml
# Edit config.yaml: set flaggems_dir, python_path, device.gpu_ids, etc.

# 2. Copy and fill in .env
cp .env.example .env
# Edit .env: set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, and optionally ANTHROPIC_MODEL

# 3. Prepare the issues list
# Edit issues_to_fix.yaml with the issues you want to fix

# 4. Run
python orchestrator.py [-v] [-c config.yaml] [issues_to_fix.yaml]
```

Options:
- `-v` / `--verbose`: Enable debug-level logging
- `-c` / `--config`: Path to config.yaml (default: `config.yaml` in script directory)

## Issues YAML Format

```yaml
issues:
  - id: 418
    operator: sparse_mla_fwd_interface
    type: accuracy_fail          # accuracy_fail | runtime_error | compilation_error | test_error
    severity: major              # optional: major | minor | unknown
    test_cmd: "pytest -m 'sparse_mla_fwd_interface' tests/ --ref cpu -vs"
    benchmark_cmd: "pytest -m 'sparse_mla_fwd_interface' benchmark/ --level core --record log"
```

Required fields: `id`, `operator`, `type`, `test_cmd`

Note: `test_cmd` should be in `pytest ...` format (without `python -m` prefix). The orchestrator template prepends `{{PYTHON_PATH}} -m` automatically.

## How It Works

1. For each issue, creates a branch `fix/issue-{id}-{operator}` in a worktree at `.worktrees/fix-{id}-{operator}`
2. Launches a Claude Code session with the `templates/fix_issue.md` prompt
3. CC reproduces the error, diagnoses root cause, implements fix, and validates
4. Results are parsed from CC's stream-json output and written to `results/summary.json`
5. On failure, retries up to `max_retries` total attempts (including the initial attempt)
6. If CC output is not parseable but the worktree has changes, the issue is marked `needs_review` for manual inspection

## Output

```
results/
├── summary.json                                    # Overall run summary
└── logs/
    ├── issue-418-sparse_mla_fwd_interface.jsonl    # Raw CC stream-json output
    ├── issue-418-sparse_mla_fwd_interface.log      # CC stderr
    └── issue-418-sparse_mla_fwd_interface.timeline.txt  # Human-readable timeline
```

### summary.json schema

```json
{
  "start_time": "2024-01-01T00:00:00+00:00",
  "end_time": "2024-01-01T01:00:00+00:00",
  "summary": {
    "total": 5,
    "success": 3,
    "failed": 1,
    "needs_review": 1,
    "in_progress": 0
  },
  "issues": {
    "issue-418": {
      "issue_id": 418,
      "operator": "sparse_mla_fwd_interface",
      "status": "success",
      "gpu_id": 0,
      "attempt": 1,
      "worktree_path": "...",
      "branch": "fix/issue-418-sparse_mla_fwd_interface",
      "duration_seconds": 600,
      "test_passed": true,
      "benchmark_passed": true,
      "format_check_passed": true,
      "cc_result": { "..." : "..." }
    }
  }
}
```

## Configuration

See `config.yaml.example` for all available options.

| Option | Default | Description |
|--------|---------|-------------|
| `flaggems_dir` | (auto-detected) | Path to FlagGems git repo |
| `python_path` | `python` | Python interpreter with FlagGems environment |
| `claude_bin` | `claude` | Claude Code executable |
| `budget_per_op` | 10000000.0 | Max budget (USD) per issue per attempt |
| `max_retries` | 2 | Total attempts per issue (including initial) |
| `timeout_per_op` | 3600 | Seconds before killing a stuck session |
| `poll_interval` | 10 | Seconds between process status checks |
| `base_branch` | master | Branch to create worktrees from |
| `gems_vendor` | nvidia | FlagGems vendor override for the environment |
| `template` | templates/fix_issue.md | Prompt template path (relative to script dir) |
| `results_dir` | results | Output directory (relative to script dir) |
| `device.lock_dir` | /tmp/auto_fix_gpu_locks | Lock file directory for GPU allocation |
| `device.gpu_ids` | null (auto-detect) | GPUs to use, e.g. `[0, 1, 2, 3]` |

## Dependencies

- Python 3.10+
- `pyyaml` (`pip install pyyaml`)
- Claude Code CLI (`claude`) installed and on PATH
- GPU environment with `nvidia-smi` available
