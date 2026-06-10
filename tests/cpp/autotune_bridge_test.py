"""Phase 3 verification: autotune_bridge end-to-end smoke test.

Runs `autotune_bridge.lookup_config` against `autotune_bridge_synth_kernel.py`
(a hand-written kernel module that combines @libtuner + @triton.heuristics +
a defaulted constexpr). Confirms that:

  1. The three sources of constexpr values (autotune / heuristics / defaults)
     are all collected.
  2. They're merged in kernel positional order, which is what the C++
     `autotuned_call` requires when forwarding `cfg.kwargs` to the kernel.
  3. bool vs int is preserved (relevant for the C++ std::variant<int64_t, bool>).
  4. Heuristic functions actually respond to args (DIVISIBLE_M flips with M).
  5. Defensive errors fire for missing function / no LibTuner in the chain.

No GPU required: the LibTuner has a single config (no benchmarking),
@triton.jit is lazy (no compilation at import), and heuristics are pure
Python lambdas.

Run:  python3 /workspace/FlagGems/tests/cpp/autotune_bridge_test.py
"""

import os
import sys
import tempfile
import textwrap

# Add libtriton_jit/scripts/ to path so `import autotune_bridge` works
# without C++ embedding setup.
SCRIPTS_DIR = "/workspace/libtriton_jit/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import autotune_bridge  # noqa: E402

SYNTH_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "autotune_bridge_synth_kernel.py",
)


def test_happy_path_three_sources_merged_in_order():
    result = autotune_bridge.lookup_config(
        kernel_path=SYNTH_KERNEL_PATH,
        function_name="synth_kernel",
        args=(None, None, 256, 128),  # A, B, M=256, N=128
        kwargs={},
    )

    assert result["num_warps"] == 4
    assert result["num_stages"] == 2

    # Order MUST match the kernel's constexpr param order (positional forward):
    #   TILE_M, TILE_N, GROUP_M  (autotune)
    #   DIVISIBLE_M, DIVISIBLE_N (heuristics)
    #   IS_FP64                  (default)
    expected_names = [
        "TILE_M",
        "TILE_N",
        "GROUP_M",
        "DIVISIBLE_M",
        "DIVISIBLE_N",
        "IS_FP64",
    ]
    assert [n for n, _ in result["kwargs"]] == expected_names

    kw = dict(result["kwargs"])
    assert kw["TILE_M"] == 128 and type(kw["TILE_M"]) is int
    assert kw["TILE_N"] == 64 and type(kw["TILE_N"]) is int
    assert kw["GROUP_M"] == 8 and type(kw["GROUP_M"]) is int
    assert kw["DIVISIBLE_M"] is True
    assert kw["DIVISIBLE_N"] is True
    assert kw["IS_FP64"] is False


def test_heuristic_responds_to_args():
    # M=200, TILE_M=128 → 200%128 != 0 → DIVISIBLE_M=False
    result = autotune_bridge.lookup_config(
        SYNTH_KERNEL_PATH,
        "synth_kernel",
        (None, None, 200, 128),
        {},
    )
    kw = dict(result["kwargs"])
    assert kw["DIVISIBLE_M"] is False
    assert kw["DIVISIBLE_N"] is True


def test_missing_function_raises():
    try:
        autotune_bridge.lookup_config(
            SYNTH_KERNEL_PATH,
            "no_such_fn",
            (),
            {},
        )
    except RuntimeError as e:
        assert "not found" in str(e)
        return
    raise AssertionError("expected RuntimeError for missing function")


def test_no_libtuner_in_chain_raises():
    bare = textwrap.dedent(
        """
        import triton
        import triton.language as tl
        @triton.jit
        def bare_kernel(A, B, M, TILE: tl.constexpr):
            pass
    """
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(bare)
        bare_path = f.name
    try:
        autotune_bridge.lookup_config(
            bare_path,
            "bare_kernel",
            (None, None, 256),
            {},
        )
    except RuntimeError as e:
        assert "no Autotuner" in str(e)
        return
    raise AssertionError("expected RuntimeError for bare @triton.jit kernel")


if __name__ == "__main__":
    test_happy_path_three_sources_merged_in_order()
    print("[1/4] happy path: three sources merged in kernel order — passed")
    test_heuristic_responds_to_args()
    print("[2/4] heuristic responds to args — passed")
    test_missing_function_raises()
    print("[3/4] missing function raises — passed")
    test_no_libtuner_in_chain_raises()
    print("[4/4] no LibTuner in chain raises — passed")
    print("autotune_bridge_test: all 4 passed")
