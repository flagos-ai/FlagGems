"""
Combination tests configuration and fixtures.

This module provides fixtures and utilities for combination testing
of FlagGems operators in realistic workflows.

NOTE: This conftest.py is designed to be compatible with the parent
tests/conftest.py. It inherits TO_CPU, QUICK_MODE, and other global
configurations.
"""

import pytest
import torch

import flag_gems

device = flag_gems.device

# Import global variables from parent conftest
try:
    from ..conftest import QUICK_MODE, RECORD_LOG, TO_CPU
except ImportError:
    # Default values if parent conftest not available
    TO_CPU = False
    QUICK_MODE = False
    RECORD_LOG = False

# ---------------------------------------------------------------------------
# Logging (initialised in pytest_configure)
# ---------------------------------------------------------------------------

_test_logger = None


# ============================================================================
# Pytest option hooks
# ============================================================================


def pytest_addoption(parser):
    """Add combination-test specific command-line options."""
    parser.addoption(
        "--combo-log-dir",
        action="store",
        default=None,
        help="Directory for combination test JSONL log files (default: combination_test_logs/)",
    )


# ============================================================================
# Register custom markers for combination tests
# ============================================================================


def pytest_configure(config):
    """Register custom markers and initialise combination-test logging."""
    # Register markers specific to combination tests
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (end-to-end scenarios)"
    )
    config.addinivalue_line(
        "markers", "numerical_stability: mark test as numerical stability test"
    )
    config.addinivalue_line("markers", "stress: mark test as stress test (may be slow)")
    config.addinivalue_line(
        "markers", "comparison: mark test as comparison test (against PyTorch baseline)"
    )
    config.addinivalue_line(
        "markers", "transformer: mark test related to Transformer models"
    )
    config.addinivalue_line(
        "markers", "attention: mark test related to attention mechanisms"
    )
    config.addinivalue_line("markers", "ffn: mark test related to FFN modules")

    # Initialise combination-test logging
    from .logging_config import TestLogger

    global _test_logger
    log_dir = config.getoption("--combo-log-dir", default=None)
    _test_logger = TestLogger(log_dir=log_dir)


# ============================================================================
# Fixtures for common test configurations
# ============================================================================


@pytest.fixture(scope="module")
def use_gems():
    """Context manager to enable FlagGems for a test module."""
    with flag_gems.use_gems():
        yield


# ============================================================================
# Pytest hooks for test lifecycle logging
# ============================================================================


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Log the start of each test with its parameters."""
    if _test_logger is None:
        return
    params = {}
    if hasattr(item, "_request") and hasattr(item._request, "node"):
        node = item._request.node
        if hasattr(node, "callspec"):
            params = {k: str(v) for k, v in node.callspec.params.items()}
    _test_logger.log_test_start(item.nodeid, params)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Log the outcome of each test (only for the 'call' phase)."""
    if _test_logger is None:
        return
    if report.when == "call":
        error_msg = None
        if report.failed and report.longrepr:
            error_msg = str(report.longrepr)
        _test_logger.log_test_result(
            test_name=report.nodeid,
            outcome=report.outcome,
            duration=report.duration,
            error_msg=error_msg,
        )
    elif report.when == "setup" and report.outcome == "skipped":
        reason = ""
        if hasattr(report.longrepr, "reprcrash"):
            reason = report.longrepr.reprcrash.message
        elif isinstance(report.longrepr, tuple):
            reason = str(report.longrepr[2])
        _test_logger.log_test_result(
            test_name=report.nodeid,
            outcome="skipped",
            duration=0,
            error_msg=reason,
        )


def pytest_sessionfinish(session, exitstatus):
    """Write session summary and close the logger."""
    if _test_logger is None:
        return
    # Gather counts from the internal _counts tracker
    passed = _test_logger._counts.get("passed", 0)
    failed = _test_logger._counts.get("failed", 0)
    skipped = _test_logger._counts.get("skipped", 0)
    total = passed + failed + skipped
    _test_logger.log_session_summary(
        total=total, passed=passed, failed=failed, skipped=skipped
    )
    _test_logger.close()


@pytest.fixture
def Float16():
    """Fixture for torch.float16 dtype."""
    return torch.float16


@pytest.fixture
def BFloat16():
    """Fixture for torch.bfloat16 dtype."""
    return torch.bfloat16


@pytest.fixture
def Float32():
    """Fixture for torch.float32 dtype."""
    return torch.float32


@pytest.fixture(
    params=[
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
    scope="module",
)
def dtype(request):
    """Parametrized fixture for common dtypes."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(1, id="batch1"),
        pytest.param(
            4, id="batch4", marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")
        ),
        pytest.param(
            16, id="batch16", marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")
        ),
    ],
    scope="module",
)
def batch_size(request):
    """Parametrized fixture for batch sizes (respects QUICK_MODE)."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(128, id="seq128"),
        pytest.param(
            512, id="seq512", marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")
        ),
        pytest.param(
            2048,
            id="seq2048",
            marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode"),
        ),
    ],
    scope="module",
)
def seq_len(request):
    """Parametrized fixture for sequence lengths (respects QUICK_MODE)."""
    return request.param


# ============================================================================
# Transformer configuration fixtures
# ============================================================================


@pytest.fixture
def transformer_config():
    """Standard Transformer configuration."""
    return {
        "d_model": 768,
        "nhead": 12,
        "dim_feedforward": 3072,
        "dropout": 0.1,
        "activation": "gelu",
    }


@pytest.fixture
def llama_config():
    """LLaMA-style Transformer configuration."""
    return {
        "d_model": 4096,
        "nhead": 32,
        "dim_feedforward": 11008,
        "dropout": 0.0,
        "activation": "silu",
        "norm": "rms_norm",
    }


# ============================================================================
# Numerical stability test utilities
# ============================================================================


@pytest.fixture
def numerical_scenarios():
    """Scenarios for numerical stability testing."""
    return {
        "normal": {
            "description": "Normal distribution inputs",
            "generator": lambda shape, dtype, device: torch.randn(
                shape, dtype=dtype, device=device
            ),
        },
        "all_neg_inf": {
            "description": "All negative infinity (padding scenario)",
            "generator": lambda shape, dtype, device: torch.full(
                shape, float("-inf"), dtype=dtype, device=device
            ),
        },
        "large_values": {
            "description": "Large values (magnitude > 10)",
            "generator": lambda shape, dtype, device: torch.randn(
                shape, dtype=dtype, device=device
            )
            * 20,
        },
        "mixed_specials": {
            "description": "Mixed normal, NaN, Inf values",
            "generator": lambda shape, dtype, device: _generate_mixed_specials(
                shape, dtype, device
            ),
        },
    }


def _generate_mixed_specials(shape, dtype, device):
    """Generate tensor with mixed normal and special values."""
    x = torch.randn(shape, dtype=dtype, device=device)
    # Inject NaN at 1% locations
    nan_mask = torch.rand(shape, device=device) < 0.01
    x[nan_mask] = float("nan")
    # Inject Inf at 0.5% locations
    inf_mask = torch.rand(shape, device=device) < 0.005
    x[inf_mask] = float("inf")
    return x


# ============================================================================
# Helper to respect QUICK_MODE in tests
# ============================================================================


def skip_if_quick_mode(reason="Skipping in QUICK_MODE"):
    """Decorator/skip marker for tests that should be skipped in quick mode."""
    return pytest.mark.skipif(QUICK_MODE, reason=reason)
