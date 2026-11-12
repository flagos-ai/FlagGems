# Hygon addmm optimization and precision guards

Isolated official-master baseline `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Hygon small/medium FP32, long-vector and BF16 kernels retain generic fallbacks. On HCU, large BF16 GEMMs use the established generic path after the specialized schedule measured severe spills/slowdowns.

## Current official-master validation

The preceding full 126-case suite is finishing sampling/accounting. A newly observed FP32 vector comparison failure is explicitly preserved in `extreme-vector-regression.json`: bias[1], A[1,3145728] stride[48,1], B[3145728,1] stride[1,48]. Old FG and DAS pass, while the preceding candidate and its independent repeat differ from the CPU FP32 reference by0.0001296997. This is a new comparison failure under the frozen tolerance, not a historical failure. The split FP64 reduction changes accumulation order; no claim that the native FP32 reference is an exact real-number oracle is made.

Current source `e42573035c2ff5a3946288542c2f0aea649241dc` retains the original generic accumulation for FP32 vector reductions with K>=2**20. **94 focused hardware tests pass**, including both ordinary and explicit-output variants of the frozen regression. All12 local pre-commit checks pass. The reference tolerance is unchanged.

`source-equivalence.json` proves that this is the only production AST difference from the preceding full suite, and evaluates the guard on every declared input. Exactly1/126 inputs changes route; the other125 preserve the complete original production AST path. The affected case has passed a complete six-role rerun, both25-sample blocks and fresh unprofiled API controls. [Full correction evidence](vector-correction-20261112/README.md) records old/new0.999911495 and DAS/new4.078290446, all output and A/A gates passed. Its entire result will replace the earlier case regardless of the resulting A/A status; no successful-repeat selection or sample pooling is allowed.

```bash
pytest tests/test_hygon_addmm.py --ref cpu -q
pre-commit run --all-files
```

This PR remains Draft until complete126-case performance accounting and latest-head CI. Tests run on Hygon gfx936, Torch2.10/Torch-FL R7, FlagTree HCU; frozen metadata-preserving generated inputs, not original model payloads. Both independent block-mean drifts must be<=5%. All producer-linked device activities are retained; API and compile costs are separate. Single-agent staged audit with independent provider processes/repeats, not independent-person approval.

## Performance guard repair in progress (2026-11-12)

The full126-case audit after the extreme-vector correction has125 new correctness passes and one inherited numerical failure. It still exposes12 accepted old/new slowdowns>5%,including7 BF16 slowdowns of approximately5–15x. These are blockers,not accepted final performance. The former backend-label guard missed the actual `hip/gfx936` compiler target.

The current candidate `0a2a9f28da8b3916c91a3b05085bca88d85c32ce` protects gfx936 as well as HCU and uses the generic route for default-scalar ragged/strided FP32 projections. An earlier unpublished guard attempt failed12 existing precision tests by also redirecting non-default alpha/beta and beta=0 reductions. The current scalar-aware guard preserves their precise path. **All108 focused hardware tests now pass** on job1102950;the same complete-output tolerances remain unchanged. Four BF16 route tests record the actual compiler target and forbid specialized launch. Full six-role performance confirmation of all16 changed metadata routes plus one predeclared unchanged control is running. Whole cases will be replaced regardless outcome;no filtered repeats or pooled samples.

The PR stays Draft until final126-case accounting and latest-head CI pass. Earlier vector-only source proof remains valid for its preceding source;the additional guarded selection is recorded separately in `performance-guard-selection.json`.
