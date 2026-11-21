# Independent official-master Hygon MM verification

Measured2026-11-12 on Hygon gfx936, Torch2.10 + Torch-FL R7 and FlagTree HCU. Tested official master `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`; isolated tested candidate `e554c140099cb3a4b216f26bc1093e7cb42a2bf1`, source tree `200959f103616b6e94e95637d254518e2b12135f`.

All **98 requested model cases plus109 declared expanded cases (207 total)** have complete old/new/DAS and repeat device timing. No old-pass/new-fail case exists across207. New full-output correctness is72/98 main and161/207 overall, versus71/98 and160/207 for both old FG and DAS. Row0140 fixes one baseline failure. The other26 main/46 overall candidate failures are inherited baseline-failing cases, retained with diagnostic timing and error details.

| Scope | Complete timing | New correct | Qualified old/new | Qualified new/DAS | Correct, stable DAS90 | Correct, stable below90 |
|---|---:|---:|---:|---:|---:|---:|
| Main98 |98|72|66|65|50|15|
| All207 |207|161|150|149|122|27|

All66 qualified main old/new pairs improve, speedups1.5750–24.9932x. All150 qualified overall pairs improve1.2591–41.6938x; no accepted slowdown over5%. Full case CSV separates point ratios from qualification flags; unqualified points are not accepted gains.

For the user's requested performance-only view, **91/98 main cases have stable new/DAS device timing regardless of numerical correctness:74 point ratios meet90%,17 are below90%;7 are A/A-inconclusive**. Only50 also satisfy the full correctness gate. All98 have timing, including26 inherited candidate numerical failures. If stability is ignored,80/98 raw point ratios meet90%; this80 count is expressly not accepted performance attainment. Main98-summary.json contains the exact counts and IDs.

The following15 fully correct/stable main comparisons remain below DAS90%. They are explicit target misses, even though all improve over official old FG:

| Case | New FG us | DAS us | DAS/New |
|---|---:|---:|---:|
| mm98-row0067 | 20.192000 | 8.280000 | 0.410063 |
| mm98-row0068 | 39.899975 | 13.424000 | 0.336441 |
| mm98-row0069 | 158.319925 | 134.127975 | 0.847196 |
| mm98-row0070 | 7.416000 | 6.160000 | 0.830636 |
| mm98-row0074 | 12.908000 | 10.240000 | 0.793306 |
| mm98-row0077 | 19.591975 | 11.472000 | 0.585546 |
| mm98-row0079 | 27.920000 | 18.795975 | 0.673208 |
| mm98-row0083 | 73.136000 | 31.032000 | 0.424305 |
| mm98-row0084 | 48.176000 | 13.896000 | 0.288442 |
| mm98-row0085 | 9.707975 | 5.608000 | 0.577669 |
| mm98-row0089 | 490.255875 | 207.555950 | 0.423362 |
| mm98-row0092 | 1876.771450 | 1290.111650 | 0.687410 |
| mm98-row0095 | 10.140000 | 6.576000 | 0.648521 |
| mm98-row0097 | 13.904000 | 7.440000 | 0.535098 |
| mm98-row0098 | 38.416000 | 18.523975 | 0.482194 |

## Measurement, source identity and limitations

Six isolated roles: old/new/DAS and independent repeats. Every role contributes2x25 samples,20 calls per ROI. Device timing sums all producer-linked kernel durations per complete ROI and divides by20. Both block-mean A/A drifts must be<=5%. API controls run separately without profiling. No sample/shard deletion, cross-run pooling or choosing a successful shard.

Hipprof can reuse a display track ID for different categories in later shards. The lossless join canonicalizes only display IDs using each shard's process label, retaining every event and its producer PID/host/device/timestamps. Full shard hashes and canonical display maps are retained; attribution checks still reject duplicate activities, missing ROIs or wrong producers. Raw binaries stay outside git.

Inputs reconstruct metadata and frozen seeds rather than original model tensor payloads. Every known stride/offset/storage/object relation is preserved; missing metadata assumptions remain explicit. Eager forward/no_grad only. The automated process-level repeats and offline audit were performed in stages by one agent; no independent-person signoff is claimed.

The published production revision differs from tested e554c140 only by one import-group blank line in mm.py; the complete src diff and full AST equivalence are verified in main98-summary.json. Test-time source fingerprints remain the measured ones rather than being silently replaced by current hashes. Historical two-case job1097237 used a legacy-FG control and **does not** establish an official-master speedup; its original receipt remains separately archived.

410 focused Hygon hardware unit tests pass, including forced FP64 banks/tails and native FP32 boundaries. Unit success is separate from the207-case performance/correctness results.

```bash
pytest tests/test_hygon_extended_mm.py tests/test_hygon_mm.py tests/test_hygon_native_fp32.py --ref cpu -q
python docs/hygon-optimization/mm/performance-20261112/verify_results.py
pre-commit run --all-files
```

## Bottom-layer execution path

Frozen CPU backing tensors transfer through Torch-FL into flagos with the declared geometry and aliases. Public `aten::mm(Tensor,Tensor)` enters exact process-private PrivateUse1 registration, then Hygon mm.py/mm_config.py and the selected native MMAC, split-K, banked or FP32 kernel. Sequential FP64 bank loading prevents overlapping live fixed-register banks; optional scheduling hints are feature-probed against the selected HCU compiler. Actual source/schema/kernel witnesses are checked for each role. Official-master FG and independent DAS boxing operate on identical effective inputs. Output transfers, wrapper/API and compile/autotune costs lie outside the kernel-sum metric. No Dynamo/Inductor fusion is active.

## Supplementary synchronized API controls

`api-controls.json` publishes all50 unprofiled samples for every old/new/DAS/new-repeat role across207 cases,with original control-file hashes and profiler-exclusion proofs. `api-case-summary.csv` places these API medians alongside the device ratios and qualification flags. The API boundary includes dispatch,allocation,launch and synchronization; it does not measure only the kernel and is not model throughput. These supplementary API points did not have a separately frozen acceptance gate; do not turn device AA qualification into an API gain claim.

Kernel gains can be smaller or absent at the complete API boundary. For main row0085(FP64[2208,64]x[64,64]),device time improves25.600->9.707975us(2.637x),while the API median is144.5755->154.0260us(old/new0.9386435;new-repeat153.8255us). For row0130(FP32[32,128]x[128,512]column-major),kernel11.892->4.172us(2.850x),API132.3855->140.4760us,new-repeat133.5405us. These are explicit API overhead/noise diagnostics,not confirmed API regression counts or a claim of end-to-end speedup. The fixed primary device acceptance counts remain unchanged.
