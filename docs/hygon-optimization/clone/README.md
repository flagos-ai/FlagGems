# Hygon `aten::clone` optimization

This branch is based on official FlagGems `master` (`4d3d9f70c9c92ae9adf004ed0202355fe26fd511`) and contains only the `clone` implementation, focused Hygon tests, and this compact review note.

## Implementation

The Hygon backend dispatches the public ATen overload to the specialized `clone` implementation and retains the generic fallback for unsupported layouts/dtypes. The source is copied from the validated internal change set `7ca605ff2692059e7e73e5bd144376264f6f5a37`; source hashes are recorded below.

## Validation status

The source change was numerically and performance tested in the internal Hygon validation on reconstructed model/expanded inputs. Before this branch is marked Ready for Review, rerun the focused test and hardware acceptance from this official-master-derived branch; no full raw profiler archive is included here.

- Focused test: `pytest tests/test_hygon_clone.py -q --ref cpu`
- Static checks: `python -m compileall -q src tests`; `git diff --check`
- Acceptance rule: each comparable case must pass full output correctness, A/A stability, and `T_DAS / T_FG >= 0.90`. Historical/common-provider failures remain diagnostic and are not counted as optimization passes.

## Source fingerprints

```json
{
  "src/flag_gems/runtime/backend/_hygon/ops/clone.py": "2ac49d3e42bed1d2618a7ea21187ff817765e78fbb4a308fb757450adf4e4724"
}
```

Full raw traces and scheduler artifacts remain in the private evidence directories under `D:/ZZ/analysis/`; they are not part of the public source change.
