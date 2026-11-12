# Hygon `aten::mm` optimization

## Scope

This isolated change adds Hygon matrix multiplication implementations and fixes FP64 MMAC operand loading, including banked and bank-tail paths. All banked paths consume one operand bank before loading the next, avoiding fixed-register overlap between live prefetch banks. Optional scheduling latency hints are enabled only when the selected compiler supports them.

## Independent validation on 2026-11-12

The tested official master baseline is `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. The tested candidate source commit is `e554c140099cb3a4b216f26bc1093e7cb42a2bf1`; documentation updates do not change those production bytes.

The complete candidate unit gate passed **410 tests, zero failures and zero errors** on Hygon hardware. It covers the workbook/expanded cases, forced FP64 bank boundaries and tails, native FP32 low-bit inputs, offsets and transposed layouts. Command:

```bash
pytest tests/test_hygon_extended_mm.py tests/test_hygon_mm.py tests/test_hygon_native_fp32.py --ref cpu -q
```

The independent performance suite is still in progress: **98 requested model cases plus 109 declared extended metadata cases**. Official-master FG, candidate FG and DAS run against identical effective inputs, with 50 device samples per role and independent repeats. Each of 50 producer-linked ROIs contains 20 calls. Only complete, correct, stable pairs count as accepted performance evidence. Historical numerical failures retain diagnostic timing. Uncaptured input metadata assumptions are disclosed per case.

No full-suite performance qualification is claimed by the unit-test result. `validation.json` records the source identity and current validation status. `historical-two-case-validation.json` preserves earlier two-case measurements: their old FG control was a legacy release, so those values do not establish improvement over official master.

## Bottom-layer execution path

The public `aten::mm(Tensor, Tensor)` call enters PrivateUse1/Torch-FL, resolves the private FG route in Hygon `ops/mm.py` and `mm_config.py`, and launches the selected Triton MMAC kernel. Hardware workers verify the executed source function and input backing/layout. The repaired FP64 lane loader feeds each bank sequentially; compiler capability checks remove unsupported optional latency hints before launch. CPU references and DAS run in independent provider processes. Device kernel duration sums, memory copies, wrapper/API costs and compile/autotune are recorded separately. Reference tolerances are unchanged.
