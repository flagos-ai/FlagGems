# Hygon advanced indexing

Guard single-axis gathers and wide/broadcast boolean compaction while retaining generic paths in slower dense/compact/reuse domains. Preserve int64 address arithmetic and negative-index wrapping in generic strided indexing. Scope is exact`aten::index.Tensor`;no unrelated operator or shared route changes.

Pinned official master`9a9aecb53103cc83dda27bc7458fc3bfd07f3188`;current tested source`e6cec3cde6cf525c5fe6e9ac93754a96d5c50a14`.49 workbook+76 declared expansions measured2026-11-12 on Hygon gfx936/Torch2.10/Torch-FL R7/FlagTree.125/125 complete samples and full output correctness for old/new/DAS;47 qualified old/new pairs,35 gains>5%,max21.101132x;55 qualified new/DAS pairs,all55DAS90. Unqualified A/A cases remain explicitly inconclusive. No newly introduced numerical or layout failure.

59 focused hardware tests pass,1 shared capability skip. All12 local static hooks passed before hardware. The final5-case guard confirmation replaces whole captures regardless outcome under the complete AST/metadata source proof. One preceding slowdown is confirmed near baseline;another retains an inconclusive0.879802x point and is not claimed fixed. See[complete evidence](performance-20261112/README.md),including every sample,actual source/ATen/kernel/backing/shard witness and date proof. No sample pooling,filtered repeats or relaxed tolerances. Inputs reconstruct known metadata and disclosed assumptions,not original model payloads.

```bash
pytest tests/test_hygon_index.py tests/test_index.py::test_index_strided_negative_indices tests/test_index.py::test_index_int32_offset_overflow --ref cpu -q
python docs/hygon-optimization/index/performance-20261112/verify_results.py
pre-commit run --all-files
```

Ready only after latest-head CI passes. Single-agent staged audit with independent provider processes/repeats,no independent-person approval claim.
