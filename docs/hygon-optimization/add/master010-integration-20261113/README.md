# ADD integration with current official Hygon master

Pinned official master: `010d8083b0dc89ab795aee973ad62c83c9d2592d`.
Measured candidate: `03e7a9e6e3f6aab80a5c925ac5653f822539c68e`.

Integrate official master010 while preserving the byte-identical ADD leaf and its existing guards. Three metadata-selected current-master controls pass full correctness and actual-provider checks for old/new/DAS. Two controls remain A/A-inconclusive; no scores from them are promoted.

72/72 focused hardware tests pass on job 1103432; JUnit and source proof
are attached. Vendor registration uses `_hygon.ops`; fully qualified imports can
create distinct objects from the same file. Export tests check actual source path
and function location, and performance workers separately trace actual calls.
No output/reference or input-geometry tolerance was relaxed.

Full200 serial-provider measurements and final accounting are pending. Actual measured source d06609e2 is preserved; leaf/config equivalence to the integrated branch and separate new-export controls are documented. This does not claim whole-package010 executed all200.

The earlier concurrent200-case run was fully timed and correct but had no accepted performance pairs because of severe process-repeat instability. Its raw arrays remain diagnostic history. A complete serial-provider200-case replacement runs all cases regardless outcome, without filtering or pooling.

Device measurements retain complete producer-linked kernel durations; API controls
are separate. Final nonproducing HIP synchronization barriers are audited without
clipping any timestamps; launches and device intervals remain exactly contained.
Nine counterexample tests cover the accounting rule. A/A threshold remains5%.
Generated frozen inputs preserve disclosed metadata,not original model payloads.
The PR remains Draft until final performance review and latest-head CI pass.
