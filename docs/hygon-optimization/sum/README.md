# Hygon sum optimization

Provide guarded short-width, long-row split and trailing-pair reductions while retaining generic fallbacks.

Independent official-master verification on2026-11-12 covers all41 declared cases. New correctness41/41, no new regressions.26 old/new pairs and27 new/DAS pairs pass independent stability gates;26 reach DAS90%. Row0649 remains below target at0.88688; other unqualified cases remain diagnostic. Qualified old/new speedups reach187.0677x with no accepted slowdown over5%.

[Per-case evidence and reproduction](performance-20261112/README.md) includes every case, failure/instability flags, input assumptions, all50 sample arrays per role, exact source fingerprints and actual kernel witnesses. This validates reconstructed metadata contracts rather than original model payload replay.
