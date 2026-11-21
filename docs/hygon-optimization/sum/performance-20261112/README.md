# Independent Hygon sum performance verification

Measured 2026-11-12 on Hygon gfx936, Torch 2.10 + Torch-FL R7 and FlagTree HCU. Official-master baseline `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`; isolated candidate `299c31ddfccc184bb1ab7a9a19be44517ae446a4`.

All **41 declared workloads** have complete timing. New FG passes full-output correctness on41/41; old FG and DAS pass40/41 each. No new correctness regressions. Among independently stable/correct pairs,26 qualify for old/new comparison and27 for new/DAS comparison.26/27 qualified DAS comparisons meet `T_DAS/T_new >=0.9`; row0649 is below target at **0.8868797308662741**. The other14 cases are not accepted DAS performance results.26 qualified old/new pairs show no slowdown over5%; the maximum accepted speedup is187.0677x. Per-case values and separate qualification flags appear in cases.csv.

Row0642 has3/360 old-FG and4/360 DAS reference mismatches; the candidate passes. Both historical failures retain diagnostic timings. Unstable A/A point ratios are diagnostic and do not count as accepted gains or passes.

The17-case scheduled confirmation replaces each selected case as a whole, irrespective of pass/fail. It does not pick the successful repeat or combine samples. `experiment_id` identifies the retained capture. Each provider and independent repeat has2x25 samples. Original captures use20calls/ROI; confirmation uses100. Both block-mean drifts must be <=5%. Device cost sums all producer-linked kernel durations, including every reduction stage, divided by calls per ROI. Compile, API time and output transfers are outside this metric. Full arrays, input backing/schema/source fingerprints and all trace-shard hashes are retained.

Inputs are reconstructed from declared metadata and frozen seeds. They are not replayed model tensor payloads; disclosed metadata assumptions remain in cases.json. Full profiler binaries stay outside the repository.

```bash
pytest tests/test_sum_hygon_short.py -q --ref cpu
python docs/hygon-optimization/sum/performance-20261112/verify_results.py
pre-commit run --all-files
```

## Bottom-layer execution path

CPU-generated backing storage moves through Torch-FL into flagos tensors, preserving declared shape/stride/offset/storage relationships. `aten::sum.dim_IntList` enters exact process-private PrivateUse1 registration, then the Hygon reduction provider. Short-width rows and trailing-pair layouts use specialized Triton/FlagTree kernels; split reductions include every device stage. Old FG uses the isolated official-master implementation; DAS uses its independent unregistered boxing process. Output is checked against the frozen CPU reference. Kernel witnesses and strict accounting-file hashes are in kernel-witness.json. No Dynamo/Inductor fusion is active.
