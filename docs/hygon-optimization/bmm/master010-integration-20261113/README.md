# BMM integration with current official Hygon master

Pinned official master: `010d8083b0dc89ab795aee973ad62c83c9d2592d`.
Measured candidate: `8ed056afe466023d1c24e6ede4060324dd73a8f1`.

Move the small-shape specialization to bmm_specialized.py so the new upstream bmm/ package cannot silently shadow it. Export that module explicitly and retain optimized official Hygon BMM as fallback. Preserve upstream out validation,resize,overlap-copy and autograd rejection behavior.

58/58 focused hardware tests pass on job 1103445; JUnit and source proof
are attached. Vendor registration uses `_hygon.ops`; fully qualified imports can
create distinct objects from the same file. Export tests check actual source path
and function location, and performance workers separately trace actual calls.
No output/reference or input-geometry tolerance was relaxed.

A fresh full32-case old/new/DAS comparison and independent repeats are running against optimized official master010. No new-baseline performance ratio or target count is accepted yet.

Earlier9a baseline results cover all32 cases with complete correctness,11 qualified old/new pairs and15 qualified new/DAS pairs. Upstream now has a genuinely different optimized BMM implementation; those historical gains do not establish gains against010.

Device measurements retain complete producer-linked kernel durations; API controls
are separate. Final nonproducing HIP synchronization barriers are audited without
clipping any timestamps; launches and device intervals remain exactly contained.
Nine counterexample tests cover the accounting rule. A/A threshold remains5%.
Generated frozen inputs preserve disclosed metadata,not original model payloads.
The PR remains Draft until final performance review and latest-head CI pass.
