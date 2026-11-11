# Hygon `aten::addmm` optimization

This branch is based on official FlagGems `master` (`4d3d9f70c9c92ae9adf004ed0202355fe26fd511`) and contains only the `addmm` implementation, focused Hygon tests, and this compact review note.

## Implementation

The Hygon backend dispatches the public ATen overload to the specialized `addmm` implementation and retains the generic fallback for unsupported layouts/dtypes. The source is copied from the validated internal change set `7ca605ff2692059e7e73e5bd144376264f6f5a37`; source hashes are recorded below.

## Validation status

The source change was numerically and performance tested in the internal Hygon validation on reconstructed model/expanded inputs. Before this branch is marked Ready for Review, rerun the focused test and hardware acceptance from this official-master-derived branch; no full raw profiler archive is included here.

- Focused test: `pytest tests/test_hygon_addmm.py -q --ref cpu`
- Static checks: `python -m compileall -q src tests`; `git diff --check`
- Acceptance rule: each comparable case must pass full output correctness, A/A stability, and `T_DAS / T_FG >= 0.90`. Historical/common-provider failures remain diagnostic and are not counted as optimization passes.

## Source fingerprints

```json
{
  "src/flag_gems/runtime/backend/_hygon/ops/addmm_kernel90.py": "61b920de4cb044177f292befa42ffd2b3b640e7a09b61670e533164318addc9a",
  "src/flag_gems/runtime/backend/_hygon/ops/addmm_schedule.py": "9027c16a8aba9f07d0163254402de8581d0467493356535412fc790df60355b0"
}
```

Full raw traces and scheduler artifacts remain in the private evidence directories under `D:/ZZ/analysis/`; they are not part of the public source change.
