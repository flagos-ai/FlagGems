# Hygon `aten::add` optimization

This branch is based on official FlagGems `master` (`4d3d9f70c9c92ae9adf004ed0202355fe26fd511`) and contains only the `add` implementation, focused Hygon tests, and this compact review note.

## Implementation

The Hygon backend dispatches the public ATen overload to the specialized `add` implementation and retains the generic fallback for unsupported layouts/dtypes. The source is copied from the validated internal change set `7ca605ff2692059e7e73e5bd144376264f6f5a37`; source hashes are recorded below.

## Validation status

The source change was numerically and performance tested in the internal Hygon validation on reconstructed model/expanded inputs. Before this branch is marked Ready for Review, rerun the focused test and hardware acceptance from this official-master-derived branch; no full raw profiler archive is included here.

- Focused test: `pytest tests/test_hygon_add.py -q --ref cpu`
- Static checks: `python -m compileall -q src tests`; `git diff --check`
- Acceptance rule: each comparable case must pass full output correctness, A/A stability, and `T_DAS / T_FG >= 0.90`. Historical/common-provider failures remain diagnostic and are not counted as optimization passes.

## Source fingerprints

```json
{
  "src/flag_gems/runtime/backend/_hygon/ops/add.py": "f8aa3867b495d3742cd3b4a239b104d936c6eb88ddb131e332bcc22800103a2d"
}
```

Full raw traces and scheduler artifacts remain in the private evidence directories under `D:/ZZ/analysis/`; they are not part of the public source change.

## Diagnostic status: no accepted performance ratios

All200 original cases and all6role samples are present. Old/new/DAS full-output correctness is200/200 for each. **Zero FG-old/new and zero new/DAS pairs qualify** because the independent process repeats fail the frozen5% block-mean stability gate. These are inconclusive measurements,not missing tests,correctness failures or accepted speedups.

For `model285-f82b55d7fcde304e3a6a`,profiled new kernel median679.607425us contrasts with the same-source independent new-repeat8.019975us;unprofiled synchronized API medians135.192us and136.057us agree much more closely. Old-repeat and DAS-repeat also show the high-duration regime. Producer-linked events name the actual add kernels rather than the excluded gate kernel. This suggests a profiling/runtime/context interaction but does not establish its root cause. The fixed-output route/timing receipts and all raw arrays remain preserved;no samples are removed.

An entire200-case confirmation now runs one provider process at a time(job1103083),with the same production source,input metadata,seeds,reference tolerance,2x25 samples/role and complete device duration boundary. Every case replaces its whole result regardless outcome;the preceding capture remains diagnostic. It changes process scheduling,not kernel arithmetic or the acceptance threshold. Final performance and Ready status remain blocked until valid evidence and latest-head CI are available. No diagnostics from this pack are written to Feishu as accepted ratios.
