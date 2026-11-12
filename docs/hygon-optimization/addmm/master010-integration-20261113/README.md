# ADDMM integration with optimized official Hygon BLAS

Pinned official master: `010d8083b0dc89ab795aee973ad62c83c9d2592d`.
Measured candidate: `bee222e93b457a9ea054860de09c8e3256637957`.

Upstream replaced its Hygon ADDMM module with a package and introduced optimized
BMM helpers. The default/out specialization now falls back to that current Hygon
implementation, preserving upstream dtype variants. The extreme FP32 vector
correction keeps the common accumulation order. Non-default scalar precision
paths remain enabled. This PR does not replace upstream BMM kernels.

All 108 focused hardware tests pass on job 1103424, with the actual HIP/gfx936
target. The JUnit and source fingerprint are attached. Local official pre-commit
rules passed; only Flake8 worker concurrency was bounded to two for Windows.

The complete 126-case old/new/DAS comparison, including all independent repeats
and API controls, is running against this new baseline. No performance data from
the earlier 9a9aecb5 baseline is claimed as gains against 010d8083.

Earlier sealed 9a results remain history: all 126 cases were timed, candidate
correctness 125/126 (the remaining failure is inherited), 42 qualified old/new
pairs, one qualified old/new slowdown of approximately 5.02%, and 62/68
qualified new/DAS pairs meeting 90%. These are not current-source acceptance.

The PR remains Draft until the new complete matrix is reviewed and latest-head
CI passes. Generated metadata-preserving inputs are not original model payloads.
