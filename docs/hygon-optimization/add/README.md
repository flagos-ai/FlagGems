# Hygon add optimization

Current guarded production source: `baa3719bd423bec0c179af709dc8404f42ca655b`,integrated on official master `010d8083b0dc89ab795aee973ad62c83c9d2592d`. Focused hardware gate: **76 tests passed** on2026-11-13. The **full200-case final performance repeat is running**;this PR remains Draft pending complete correctness,device accounting,stability review and latest-head required CI.

## Why the additional guard and full repeat

The complete preceding serial200 capture has200/200 old/new/DAS correctness,79 qualified old/new pairs,44 gains>5%,and3 qualified device regressions: large row broadcast0.859160x,INT64 vector0.945983x,and an intermediate transpose0.603094x. Synchronized API controls improve for all three,but do not establish device-kernel acceptance.99 new/DAS pairs qualify,75 meet90%;other comparisons retain A/A diagnostics. The complete [preceding serial capture](diagnostic-serial-20261113/README.md) remains auditable,as does [the original concurrent-process diagnostic](diagnostic-concurrent-20261112/README.md).

The guard uses general dtype,volume,stride and broadcast domains,selecting exactly3 of the frozen200 workload metadata records. All other production-module AST is unchanged. Every200 case is being rerun on the pinned optimized master with fresh six-role captures and API controls,not just the three regressed shapes. No previous samples are pooled or selected by outcome. The previous9a ADD leaf/config equivalence and three actual current-master route controls are historical support,not a claim that an entire9a execution used the010 package.

## Reproduction and actual execution

```bash
pytest tests/test_hygon_add.py -q --ref cpu
python docs/hygon-optimization/add/diagnostic-serial-20261113/verify_results.py
python docs/hygon-optimization/add/diagnostic-serial-20261113/test-accounting-boundaries.py
pre-commit run --all-files
```

Public `aten::add.Tensor` uses an exact private PrivateUse1 binding to the real candidate source. The provider selects contiguous/ordered/transpose kernels or generic pointwise fallback. Full output and input backing,known alias,stride,offset and immutability are checked outside the ROI. Inputs reconstruct fixed generated metadata contracts with explicit assumptions,not original model tensor payloads. Six sequential provider roles each have2x25 valid device ROIs and20calls/ROI,with independent block drift<=5%. Device metric is the complete producer-linked kernel-duration sum. Separate50-call synchronized API controls include CPU dispatch,allocation,launch and synchronization and are not model throughput. Eager execution,no graph fusion claim.

The V5 accountant records a final same-thread nonproducing HIP wait overrun separately while preserving exact producing-API and device containment. All events and timestamps remain;no epsilon,trimming or sample exclusion. Exact source and9 boundary counterexamples are included. Hygon gfx936,Torch2.10/Torch-FL R7,FlagTree HIP/gfx936. Single-agent staged audit with independent provider processes/repeats,no independent-person signoff claim.
