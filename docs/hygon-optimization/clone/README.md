# Hygon clone optimization

Guarded Hygon copy kernels retain generic fallback, including large FP64 noncontiguous-to-contiguous copies that regressed in the initial candidate.

Independent official-master verification2026-11-12: full32/32 correctness and timing;10 focused hardware units passed.18 stable/correct old/new pairs qualify;11 improve over5%, maximum1.7137x, none slows over5%.22 new/DAS pairs qualify,17 meet90%; five misses are explicitly listed. Unstable pairs remain diagnostic.

[Per-case evidence and reproduction](performance-20261112/README.md) includes complete kernel-plus-copy timing, all50 samples per role, input assumptions, production/source/shard hashes, actual kernel/copy witnesses and offline verification. Inputs reconstruct metadata and seeds rather than replay original model payloads.
