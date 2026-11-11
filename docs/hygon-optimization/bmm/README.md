# Hygon `aten::bmm` optimization

This branch is based on official FlagGems `master` (`4d3d9f70c9c92ae9adf004ed0202355fe26fd511`) and contains only the `bmm` implementation, focused Hygon tests, and this compact review note.

## Implementation

The Hygon backend dispatches the public ATen overload to the specialized `bmm` implementation and retains the generic fallback for unsupported layouts/dtypes. The source is copied from the validated internal change set `7ca605ff2692059e7e73e5bd144376264f6f5a37`; source hashes are recorded below.

## Validation status

The source change was numerically and performance tested in the internal Hygon validation on reconstructed model/expanded inputs. Before this branch is marked Ready for Review, rerun the focused test and hardware acceptance from this official-master-derived branch; no full raw profiler archive is included here.

- Focused test: `pytest tests/test_hygon_bmm.py -q --ref cpu`
- Static checks: `python -m compileall -q src tests`; `git diff --check`
- Acceptance rule: each comparable case must pass full output correctness, A/A stability, and `T_DAS / T_FG >= 0.90`. Historical/common-provider failures remain diagnostic and are not counted as optimization passes.

## Source fingerprints

```json
{
  "src/flag_gems/runtime/backend/_hygon/ops/bmm.py": "37c4595b39504e7a820ec03f5c2195c811dd9a07563627edcca011677e313d48"
}
```

Full raw traces and scheduler artifacts remain in the private evidence directories under `D:/ZZ/analysis/`; they are not part of the public source change.
