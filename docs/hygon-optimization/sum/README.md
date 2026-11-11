# Hygon `aten::sum` optimization

Based on official FlagGems `master` (`4d3d9f70c9c92ae9adf004ed0202355fe26fd511`). Add short-width, long-row split, and trailing-pair reduction paths while preserving shape, dtype, keepdim, offset, and generic fallback semantics. Internal validation covered all 32 requested workloads plus controls.

Before Ready for Review, rerun the focused tests and hardware acceptance from this official-master-derived branch. Every comparable case must pass correctness, A/A stability, and `T_DAS / T_FG >= 0.90`. Raw profiler/scheduler archives remain private.

Source SHA256: `42e65f32a2c471acd1c273366c708689e85fe668eb148b4950d7fef5f9725b5a`
