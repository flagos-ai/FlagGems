# Hygon `aten::index.Tensor` optimization

Isolated branch based on official master `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Specialized gather and boolean-compaction kernels retain generic fallbacks. All offset products use int64 addressing; the generic negative-index wrap is applied after promotion as well.

## Current hardware validation

On 2026-11-12, the preceding source completed all125 declared model/expanded cases with old/new/DAS correctness passing125/125. Four stable old/new device comparisons regressed by more than5%; these results remain diagnostic and are not final evidence for the current guard.

The current source `1a5cea0e25316bb5159ccefcbea1245542a20b2a` retains generic indexing for contiguous dense tiny tails, large outputs and high-reuse INT64 scalar gathers. The guards depend on sizes/layouts, not workload IDs or input values. **55 focused hardware tests passed, with1 shared memory-capability test skipped.** Dedicated Hygon tests really execute positive/negative int32/int64 gathers above2GiB address offsets. Fourteen new full-output tests cover repeated negative and positive boundary rows in the guarded regimes.

The complete125-case repeat is in progress. This PR remains Draft until final source-bound performance accounting and latest-head CI finish. No claim that the current candidate already achieves a125-case performance pass rate is made.

```bash
pytest tests/test_hygon_index.py tests/test_index.py::test_index_strided_negative_indices tests/test_index.py::test_index_int32_offset_overflow --ref cpu -q
pre-commit run --all-files
```

All12 local pre-commit checks pass. Each old/new/DAS provider and independent repeat uses2x25 complete producer-linked device samples,20 calls/ROI. Output correctness and per-block A/A drift<=5% qualify pairs separately. Full reconstructed input backing/geometry and known aliases are preserved; these are generated values, not original model payload replay. Large trace-shard re-accounting retains every event and original producer/timestamp identity; it never substitutes a partial trace median.

## Source fingerprints

```json
{
  "src/flag_gems/runtime/backend/_hygon/ops/index.py": "5c698ef1295676de5694de8c6312c65c93603bdf1560548006fa18404797a40a",
  "src/flag_gems/ops/index.py": "3c0a6b86235f18988d6581350a421bf99641067683786a0648146c9747213705"
}
```

Compact complete-case evidence will be added after the full repeat. Large raw/profiler binaries remain outside the public repository. Single-agent staged audit with separate provider processes and repeats; no independent-person signoff claimed.

## Compact-row follow-up (2026-11-12)

The preceding source `1a5cea0e25316bb5159ccefcbea1245542a20b2a` completed all125 cases and all6role device samples. Every provider passes correctness125/125;57 stable/correct new/DAS pairs all meet90%. Among49 qualified old/new comparisons,38 improve>5%,but2 expanded cases still regress:singleton[1,1] with31360 indices(old/new0.876812) and[913,3,24] with2101 indices(0.909097). These are retained performance blockers,not final accepted results.

Current source `e6cec3cde6cf525c5fe6e9ac93754a96d5c50a14` retains the generic path for compact rows and highly reused singleton gathers. **59 focused hardware tests pass,1 shared capability skip**,including4 new complete-output and explicit-generic-route tests. An entire-module AST comparison and evaluation of all125 frozen metadata inputs identifies exactly5 affected cases. All5 are being rerun with full six-role samples,regardless final correctness or stability outcome. Unchanged cases can be retained only under this source proof;no sample pooling or successful-repeat selection. `compact-guard-selection.json` records the premeasurement selection. Latest-head CI and final combined evidence remain required before Ready.
