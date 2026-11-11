# Hygon `_scaled_dot_product_flash_attention` optimization

Based on official FlagGems `master` (`4d3d9f70c9c92ae9adf004ed0202355fe26fd511`). Add guarded Hygon paths for the four reviewed BF16 shared-backing layouts while preserving output/LSE semantics and stable fallbacks. Internal validation used native 2.8.4 comparison and full output/LSE stability checks. Before Ready for Review, rerun focused tests and hardware acceptance from this official-master-derived branch.

Source hashes:
```json
{
  "src/flag_gems/runtime/backend/_hygon/attention_state_hip.py": "185460a1f7a25f27ba55202cf42f35fef5a10fda845b527c3058b5b9b88f12d4",
  "src/flag_gems/runtime/backend/_hygon/ops/attention_state.py": "156b334c1a07150624f6eff2f8b059f7fef2f209b1d02e63235c7ac032040213",
  "src/flag_gems/runtime/backend/_hygon/ops/_flash_attention_forward.py": "1be5bfed085bc45c9628da0c4904c29fc9de619517353bbb1b000bc0a5e77ceb",
  "src/flag_gems/runtime/backend/_hygon/ops/_scaled_dot_product_flash_attention.py": "6eabe38ca1abcf6ff68c30b40250895b45f0fb402c6acad82bb50982b4ac3a0e"
}
```
