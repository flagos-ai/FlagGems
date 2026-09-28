#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_test_result.py - Compare Two Test Result Summaries

Compares two test result summary.json files, identifies differences,
regressions, improvements, and other changes. Generates both human-readable
and JSON reports.

Usage:
    python compare_test_result.py --baseline baseline/summary.json --candidate candidate/summary.json
"""

import argparse
import datetime
import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

# ============================================================================
# Data Models
# ============================================================================


class ChangeType(Enum):
    """Types of status changes between baseline and candidate"""

    # Critical regressions (block merge)
    PASSED_TO_FAILED = "PASSED_TO_FAILED"
    PASSED_TO_ERROR = "PASSED_TO_ERROR"
    SKIPPED_TO_FAILED = "SKIPPED_TO_FAILED"
    NOT_FOUND_TO_FAILED = "NOT_FOUND_TO_FAILED"
    NOT_FOUND_TO_ERROR = "NOT_FOUND_TO_ERROR"

    # Improvements (good)
    FAILED_TO_PASSED = "FAILED_TO_PASSED"
    ERROR_TO_PASSED = "ERROR_TO_PASSED"
    SKIPPED_TO_PASSED = "SKIPPED_TO_PASSED"
    NOT_FOUND_TO_PASSED = "NOT_FOUND_TO_PASSED"

    # Warnings (caution)
    PASSED_TO_SKIPPED = "PASSED_TO_SKIPPED"
    PASSED_TO_NOT_FOUND = "PASSED_TO_NOT_FOUND"
    FAILED_TO_ERROR = "FAILED_TO_ERROR"
    ERROR_TO_FAILED = "ERROR_TO_FAILED"
    NOT_FOUND_TO_SKIPPED = "NOT_FOUND_TO_SKIPPED"

    # Info (informational)
    FAILED_TO_SKIPPED = "FAILED_TO_SKIPPED"
    SKIPPED_TO_SKIPPED = "SKIPPED_TO_SKIPPED"
    FAILED_TO_FAILED = "FAILED_TO_FAILED"
    ERROR_TO_ERROR = "ERROR_TO_ERROR"

    # Unchanged
    UNCHANGED = "UNCHANGED"


class Severity(Enum):
    """Severity levels for changes"""

    CRITICAL = "critical"  # Blocks merge
    GOOD = "good"  # Improvement
    WARNING = "warning"  # Needs attention
    INFO = "info"  # Informational only
    UNCHANGED = "unchanged"  # No change


# Severity mapping for each change type
CHANGE_SEVERITY = {
    ChangeType.PASSED_TO_FAILED: Severity.CRITICAL,
    ChangeType.PASSED_TO_ERROR: Severity.CRITICAL,
    ChangeType.SKIPPED_TO_FAILED: Severity.CRITICAL,
    ChangeType.NOT_FOUND_TO_FAILED: Severity.CRITICAL,
    ChangeType.NOT_FOUND_TO_ERROR: Severity.CRITICAL,
    ChangeType.FAILED_TO_PASSED: Severity.GOOD,
    ChangeType.ERROR_TO_PASSED: Severity.GOOD,
    ChangeType.SKIPPED_TO_PASSED: Severity.GOOD,
    ChangeType.NOT_FOUND_TO_PASSED: Severity.GOOD,
    ChangeType.PASSED_TO_SKIPPED: Severity.WARNING,
    ChangeType.PASSED_TO_NOT_FOUND: Severity.WARNING,
    ChangeType.FAILED_TO_ERROR: Severity.WARNING,
    ChangeType.ERROR_TO_FAILED: Severity.WARNING,
    ChangeType.NOT_FOUND_TO_SKIPPED: Severity.WARNING,
    ChangeType.FAILED_TO_SKIPPED: Severity.INFO,
    ChangeType.SKIPPED_TO_SKIPPED: Severity.INFO,
    ChangeType.FAILED_TO_FAILED: Severity.INFO,
    ChangeType.ERROR_TO_ERROR: Severity.INFO,
    ChangeType.UNCHANGED: Severity.UNCHANGED,
}

# Human-readable labels for change types
CHANGE_LABELS = {
    ChangeType.PASSED_TO_FAILED: "Passed → Failed",
    ChangeType.PASSED_TO_ERROR: "Passed → Error",
    ChangeType.SKIPPED_TO_FAILED: "Skipped → Failed",
    ChangeType.NOT_FOUND_TO_FAILED: "New Test Failed",
    ChangeType.NOT_FOUND_TO_ERROR: "New Test Error",
    ChangeType.FAILED_TO_PASSED: "Failed → Passed",
    ChangeType.ERROR_TO_PASSED: "Error → Passed",
    ChangeType.SKIPPED_TO_PASSED: "Skipped → Passed",
    ChangeType.NOT_FOUND_TO_PASSED: "New Test Passed",
    ChangeType.PASSED_TO_SKIPPED: "Passed → Skipped",
    ChangeType.PASSED_TO_NOT_FOUND: "Test Removed",
    ChangeType.FAILED_TO_ERROR: "Failed → Error",
    ChangeType.ERROR_TO_FAILED: "Error → Failed",
    ChangeType.NOT_FOUND_TO_SKIPPED: "New Test Skipped",
    ChangeType.FAILED_TO_SKIPPED: "Failed → Skipped",
    ChangeType.SKIPPED_TO_SKIPPED: "Skipped → Skipped",
    ChangeType.FAILED_TO_FAILED: "Still Failed",
    ChangeType.ERROR_TO_ERROR: "Still Error",
    ChangeType.UNCHANGED: "Unchanged",
}


@dataclass
class TestCaseResult:
    """Represents a single test case result"""

    test_name: str
    status: str  # PASSED, FAILED, SKIPPED, ERROR, NOT_FOUND


@dataclass
class OperatorResult:
    """Represents results for a single operator"""

    name: str
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    status: str
    passed_cases: List[str] = field(default_factory=list)
    failed_cases: List[str] = field(default_factory=list)
    skipped_cases: List[str] = field(default_factory=list)
    error_cases: List[str] = field(default_factory=list)


@dataclass
class TestSummary:
    """Complete test summary"""

    timestamp: str
    duration_seconds: float
    operators: Dict[str, OperatorResult]
    totals: Dict[str, int]
    skip_lists: Optional[Dict] = None


@dataclass
class Change:
    """A single change between baseline and candidate"""

    operator: str
    test_case: str
    baseline_status: str
    candidate_status: str
    change_type: ChangeType
    severity: Severity


@dataclass
class ComparisonReport:
    """Complete comparison report"""

    timestamp: str
    baseline_path: str
    candidate_path: str
    baseline_label: str
    candidate_label: str

    # Statistics
    statistics: Dict
    totals: Dict

    # Changes categorized
    regressions: List[Change] = field(default_factory=list)
    improvements: List[Change] = field(default_factory=list)
    warnings: List[Change] = field(default_factory=list)
    info_changes: List[Change] = field(default_factory=list)

    # Operator-level changes
    new_operators: List[str] = field(default_factory=list)
    removed_operators: List[str] = field(default_factory=list)

    # Summary counts
    summary: Dict = field(default_factory=dict)


# ============================================================================
# Core Comparison Logic
# ============================================================================


class TestComparator:
    """Compare two test summaries"""

    def __init__(
        self,
        baseline_path: Path,
        candidate_path: Path,
        baseline_label: Optional[str] = None,
        candidate_label: Optional[str] = None,
    ):
        self.baseline_path = baseline_path
        self.candidate_path = candidate_path
        self.baseline_label = baseline_label or "Baseline"
        self.candidate_label = candidate_label or "Candidate"
        self.baseline: Optional[TestSummary] = None
        self.candidate: Optional[TestSummary] = None
        self.report: Optional[ComparisonReport] = None

    def load_summaries(self) -> bool:
        """Load both summary files"""
        try:
            with open(self.baseline_path, "r") as f:
                baseline_data = json.load(f)

            with open(self.candidate_path, "r") as f:
                candidate_data = json.load(f)

            self.baseline = self._parse_summary(baseline_data, self.baseline_path)
            self.candidate = self._parse_summary(candidate_data, self.candidate_path)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load summaries: {e}")
            return False

    def _parse_summary(self, data: Dict, path: Path) -> TestSummary:
        """Parse summary JSON into structured data"""
        operators = {}
        for op_name, op_data in data.get("operators", {}).items():
            operators[op_name] = OperatorResult(
                name=op_name,
                total=op_data.get("total", 0),
                passed=op_data.get("passed", 0),
                failed=op_data.get("failed", 0),
                skipped=op_data.get("skipped", 0),
                errors=op_data.get("errors", 0),
                status=op_data.get("status", "Unknown"),
                passed_cases=op_data.get("passed_cases", []),
                failed_cases=op_data.get("failed_cases", []),
                skipped_cases=op_data.get("skipped_cases", []),
                error_cases=op_data.get("error_cases", []),
            )

        return TestSummary(
            timestamp=data.get("timestamp", ""),
            duration_seconds=data.get("duration_seconds", 0),
            operators=operators,
            totals=data.get("totals", {}),
            skip_lists=data.get("skip_lists", {}),
        )

    def _get_test_status(
        self, operator_result: Optional[OperatorResult], test_case: str
    ) -> str:
        """Get the status of a specific test case"""
        if operator_result is None:
            return "NOT_FOUND"

        if test_case in operator_result.passed_cases:
            return "PASSED"
        elif test_case in operator_result.failed_cases:
            return "FAILED"
        elif test_case in operator_result.skipped_cases:
            return "SKIPPED"
        elif test_case in operator_result.error_cases:
            return "ERROR"
        return "NOT_FOUND"

    def _get_all_test_cases(self, operator: str) -> Set[str]:
        """Get all test cases for an operator from both baseline and candidate"""
        cases = set()

        # From baseline
        if operator in self.baseline.operators:
            op = self.baseline.operators[operator]
            cases.update(op.passed_cases)
            cases.update(op.failed_cases)
            cases.update(op.skipped_cases)
            cases.update(op.error_cases)

        # From candidate
        if operator in self.candidate.operators:
            op = self.candidate.operators[operator]
            cases.update(op.passed_cases)
            cases.update(op.failed_cases)
            cases.update(op.skipped_cases)
            cases.update(op.error_cases)

        return cases

    def _classify_change(
        self, baseline_status: str, candidate_status: str
    ) -> ChangeType:
        """Classify the change between two statuses"""
        key = f"{baseline_status}_TO_{candidate_status}"
        try:
            return ChangeType(key)
        except ValueError:
            return ChangeType.UNCHANGED

    def compare(self) -> ComparisonReport:
        """Execute the comparison"""
        all_operators = set(self.baseline.operators.keys()) | set(
            self.candidate.operators.keys()
        )

        report = ComparisonReport(
            timestamp=datetime.datetime.now().isoformat(),
            baseline_path=str(self.baseline_path),
            candidate_path=str(self.candidate_path),
            baseline_label=self.baseline_label,
            candidate_label=self.candidate_label,
            statistics={},
            totals={},
        )

        # Track statistics
        baseline_total_cases = 0
        candidate_total_cases = 0
        baseline_passed = 0
        candidate_passed = 0
        baseline_failed = 0
        candidate_failed = 0
        baseline_skipped = 0
        candidate_skipped = 0
        baseline_errors = 0
        candidate_errors = 0

        # Compare each operator
        for op in sorted(all_operators):
            baseline_op = self.baseline.operators.get(op)
            candidate_op = self.candidate.operators.get(op)

            # Track new/removed operators
            if baseline_op is None and candidate_op is not None:
                report.new_operators.append(op)
            elif baseline_op is not None and candidate_op is None:
                report.removed_operators.append(op)

            # Get all test cases for this operator
            all_cases = self._get_all_test_cases(op)

            for test_case in all_cases:
                baseline_status = self._get_test_status(baseline_op, test_case)
                candidate_status = self._get_test_status(candidate_op, test_case)

                # Update statistics
                if baseline_status == "PASSED":
                    baseline_passed += 1
                elif baseline_status == "FAILED":
                    baseline_failed += 1
                elif baseline_status == "SKIPPED":
                    baseline_skipped += 1
                elif baseline_status == "ERROR":
                    baseline_errors += 1
                baseline_total_cases += 1

                if candidate_status == "PASSED":
                    candidate_passed += 1
                elif candidate_status == "FAILED":
                    candidate_failed += 1
                elif candidate_status == "SKIPPED":
                    candidate_skipped += 1
                elif candidate_status == "ERROR":
                    candidate_errors += 1
                candidate_total_cases += 1

                # Skip unchanged
                if baseline_status == candidate_status:
                    continue

                # Classify the change
                change_type = self._classify_change(baseline_status, candidate_status)
                severity = CHANGE_SEVERITY.get(change_type, Severity.UNCHANGED)

                change = Change(
                    operator=op,
                    test_case=test_case,
                    baseline_status=baseline_status,
                    candidate_status=candidate_status,
                    change_type=change_type,
                    severity=severity,
                )

                # Add to appropriate category
                if severity == Severity.CRITICAL:
                    report.regressions.append(change)
                elif severity == Severity.GOOD:
                    report.improvements.append(change)
                elif severity == Severity.WARNING:
                    report.warnings.append(change)
                elif severity == Severity.INFO:
                    report.info_changes.append(change)

        # Save statistics
        report.statistics = {
            "operators": {
                "baseline": len(self.baseline.operators),
                "candidate": len(self.candidate.operators),
                "delta": len(self.candidate.operators) - len(self.baseline.operators),
            },
            "test_cases": {
                "baseline": baseline_total_cases,
                "candidate": candidate_total_cases,
                "delta": candidate_total_cases - baseline_total_cases,
            },
            "passed": {
                "baseline": baseline_passed,
                "candidate": candidate_passed,
                "delta": candidate_passed - baseline_passed,
            },
            "failed": {
                "baseline": baseline_failed,
                "candidate": candidate_failed,
                "delta": candidate_failed - baseline_failed,
            },
            "skipped": {
                "baseline": baseline_skipped,
                "candidate": candidate_skipped,
                "delta": candidate_skipped - baseline_skipped,
            },
            "errors": {
                "baseline": baseline_errors,
                "candidate": candidate_errors,
                "delta": candidate_errors - baseline_errors,
            },
        }

        # Calculate totals
        report.totals = {
            "total_operators": len(all_operators),
            "new_operators": len(report.new_operators),
            "removed_operators": len(report.removed_operators),
            "regressions": len(report.regressions),
            "improvements": len(report.improvements),
            "warnings": len(report.warnings),
            "info_changes": len(report.info_changes),
        }

        # Summary for quick reference
        report.summary = {
            "has_critical_changes": len(report.regressions) > 0,
            "has_improvements": len(report.improvements) > 0,
            "has_warnings": len(report.warnings) > 0,
            "recommendation": (
                "❌ REJECT" if len(report.regressions) > 0 else "✅ ACCEPT"
            ),
        }

        self.report = report
        return report


# ============================================================================
# Report Generators
# ============================================================================


class ReportGenerator:
    """Generate reports in various formats"""

    @staticmethod
    def to_json(report: ComparisonReport, pretty: bool = True) -> str:
        """Convert report to JSON"""

        def serialize(obj):
            if isinstance(obj, Change):
                return {
                    "operator": obj.operator,
                    "test_case": obj.test_case,
                    "baseline_status": obj.baseline_status,
                    "candidate_status": obj.candidate_status,
                    "change_type": obj.change_type.value,
                    "severity": obj.severity.value,
                }
            return obj

        return json.dumps(
            asdict(report, dict_factory=lambda x: {k: serialize(v) for k, v in x}),
            indent=2 if pretty else None,
            default=str,
        )

    @staticmethod
    def to_markdown(report: ComparisonReport) -> str:
        """Convert report to Markdown"""
        lines = []

        # Header
        lines.append("## 🧪 Test Comparison Report")
        lines.append(f"\n**Baseline**: `{report.baseline_path}`")
        lines.append(f"**Candidate**: `{report.candidate_path}`")
        lines.append(f"**Report Time**: {report.timestamp}")

        # Overall Summary
        lines.append("\n### 📊 Overall Summary")
        lines.append("| Metric | Baseline | Candidate | Delta |")
        lines.append("|--------|----------|-----------|-------|")

        stats = report.statistics
        lines.append(
            f"  Total Operators  |  {stats['operators']['baseline']}  "
            f"|  {stats['operators']['candidate']}  "
            f"|  {stats['operators']['delta']:+d}  |"
        )
        lines.append(
            f"  Total Test Cases  |  {stats['test_cases']['baseline']}  "
            f"|  {stats['test_cases']['candidate']}  "
            f"|  {stats['test_cases']['delta']:+d}  |"
        )
        lines.append(
            f"  ✅ Passed  |  {stats['passed']['baseline']}  "
            f"|  {stats['passed']['candidate']}  "
            f"|  {stats['passed']['delta']:+d}  |"
        )
        lines.append(
            f"  ❌ Failed  |  {stats['failed']['baseline']}  "
            f"|  {stats['failed']['candidate']}  "
            f"|  {stats['failed']['delta']:+d}  |"
        )
        lines.append(
            f"  ⏭️ Skipped  |  {stats['skipped']['baseline']}  "
            f"|  {stats['skipped']['candidate']}  "
            f"|  {stats['skipped']['delta']:+d}  |"
        )
        lines.append(
            f"  ⚠ Errors  |  {stats['errors']['baseline']}  "
            f"|  {stats['errors']['candidate']}  "
            f"|  {stats['errors']['delta']:+d}  |"
        )

        # Changes Summary
        lines.append("\n### 🔄 Changes Summary")
        lines.append(f"- 🔴 **Regressions**: {len(report.regressions)}")
        lines.append(f"- 🟢 **Improvements**: {len(report.improvements)}")
        lines.append(f"- 🟡 **Warnings**: {len(report.warnings)}")
        lines.append(f"- ℹ️ **Info Changes**: {len(report.info_changes)}")

        # New/Removed Operators
        if report.new_operators:
            lines.append(f"\n### 🆕 New Operators ({len(report.new_operators)})")
            for op in report.new_operators[:10]:
                lines.append(f"- `{op}`")
            if len(report.new_operators) > 10:
                lines.append(f"- ... and {len(report.new_operators) - 10} more")

        if report.removed_operators:
            lines.append(
                f"\n### ❌ Removed Operators ({len(report.removed_operators)})"
            )
            for op in report.removed_operators[:10]:
                lines.append(f"- `{op}`")
            if len(report.removed_operators) > 10:
                lines.append(f"- ... and {len(report.removed_operators) - 10} more")

        # Regressions (Critical)
        if report.regressions:
            lines.append(f"\n### 🔴 Critical Regressions ({len(report.regressions)})")
            lines.append("| Operator | Test Case | Baseline | Candidate |")
            lines.append("|----------|-----------|----------|-----------|")
            for change in report.regressions[:20]:
                lines.append(
                    f"| `{change.operator}` | `{change.test_case}` |"
                    f" {change.baseline_status} | {change.candidate_status} |"
                )
            if len(report.regressions) > 20:
                lines.append("| ... | ... | ... | ... |")
                lines.append(
                    f"| **Total** | **{len(report.regressions)} regressions** | | |"
                )

        # Improvements
        if report.improvements:
            lines.append(f"\n### 🟢 Improvements ({len(report.improvements)})")
            lines.append("| Operator | Test Case | Baseline | Candidate |")
            lines.append("|----------|-----------|----------|-----------|")
            for change in report.improvements[:20]:
                lines.append(
                    f"| `{change.operator}` | `{change.test_case}` |"
                    f" {change.baseline_status} | {change.candidate_status} |"
                )
            if len(report.improvements) > 20:
                lines.append("| ... | ... | ... | ... |")
                lines.append(
                    f"| **Total** | **{len(report.improvements)} improvements** | | |"
                )

        # Warnings
        if report.warnings:
            lines.append(f"\n### 🟡 Warnings ({len(report.warnings)})")
            lines.append("| Operator | Test Case | Change |")
            lines.append("|----------|-----------|--------|")
            for change in report.warnings[:10]:
                label = CHANGE_LABELS.get(change.change_type, change.change_type.value)
                lines.append(
                    f"| `{change.operator}` | `{change.test_case}` | {label} |"
                )
            if len(report.warnings) > 10:
                lines.append("| ... | ... | ... |")
                lines.append(f"| **Total** | **{len(report.warnings)} warnings** | |")

        # Recommendation
        lines.append("\n### 📝 Recommendation")
        if report.summary["has_critical_changes"]:
            lines.append(
                "❌ **REJECT** - Critical regressions detected. Please fix before proceeding."
            )
        elif report.summary["has_warnings"]:
            lines.append(
                "⚠️ **REVIEW REQUIRED** - Warnings detected. Please review before proceeding."
            )
        else:
            lines.append("✅ **ACCEPT** - No critical issues detected.")

        return "\n".join(lines)

    @staticmethod
    def to_console(report: ComparisonReport) -> str:
        """Convert report to console-friendly text with colors"""
        try:
            import colorama

            colorama.init()
            has_color = True
        except ImportError:
            has_color = False

        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("                    Test Comparison Report")
        lines.append("=" * 80)
        lines.append(f"Baseline: {report.baseline_path}")
        lines.append(f"Candidate: {report.candidate_path}")
        lines.append("")

        # Overall Statistics
        lines.append("📊 Overall Statistics")
        lines.append("─" * 80)
        lines.append(f"{'Metric':<20} {'Baseline':>12} {'Candidate':>12} {'Delta':>10}")
        lines.append("─" * 80)

        stats = report.statistics
        lines.append(
            f"{'Total Operators':<20} "
            f"{stats['operators']['baseline']:>12} "
            f"{stats['operators']['candidate']:>12} "
            f"{stats['operators']['delta']:>+10}"
        )
        lines.append(
            f"{'Total Test Cases':<20} "
            f"{stats['test_cases']['baseline']:>12} "
            f"{stats['test_cases']['candidate']:>12} "
            f"{stats['test_cases']['delta']:>+10}"
        )

        if has_color:

            def color_delta(delta):
                if delta > 0:
                    return f"\033[32m{delta:+d}\033[0m"
                elif delta < 0:
                    return f"\033[31m{delta:+d}\033[0m"
                return f"{delta:+d}"

        else:

            def color_delta(delta):
                return f"{delta:+d}"

        # Format passed with percentage
        baseline_total = stats["test_cases"]["baseline"]
        candidate_total = stats["test_cases"]["candidate"]

        if baseline_total > 0:
            baseline_pass_rate = (
                f"({stats['passed']['baseline'] * 100 / baseline_total:.1f}%)"
            )
        else:
            baseline_pass_rate = "(0.0%)"

        if candidate_total > 0:
            candidate_pass_rate = (
                f"({stats['passed']['candidate'] * 100 / candidate_total:.1f}%)"
            )
        else:
            candidate_pass_rate = "(0.0%)"

        lines.append(
            f"{'✅ Passed':<20} "
            f"{stats['passed']['baseline']:>8} "
            f"{baseline_pass_rate:>6} "
            f"{stats['passed']['candidate']:>8} "
            f"{candidate_pass_rate:>6} "
            f"{color_delta(stats['passed']['delta']):>10}"
        )
        lines.append(
            f"{'❌ Failed':<20} "
            f"{stats['failed']['baseline']:>12} "
            f"{stats['failed']['candidate']:>12} "
            f"{color_delta(stats['failed']['delta']):>10}"
        )
        lines.append(
            f"{'⏭ Skipped':<20} "
            f"{stats['skipped']['baseline']:>12} "
            f"{stats['skipped']['candidate']:>12} "
            f"{color_delta(stats['skipped']['delta']):>10}"
        )
        lines.append(
            f"{'⚠ Errors':<20} "
            f"{stats['errors']['baseline']:>12} "
            f"{stats['errors']['candidate']:>12} "
            f"{color_delta(stats['errors']['delta']):>10}"
        )
        lines.append("─" * 80)

        # Changes Summary
        lines.append("")
        lines.append("🔄 Changes Summary")
        lines.append("─" * 80)

        if has_color:

            def color_severity(count, emoji, color):
                if count > 0:
                    return f"{color}{emoji} {count}{colorama.Style.RESET_ALL}"
                return f"{emoji} {count}"

        else:

            def color_severity(count, emoji, color):
                return f"{emoji} {count}"

        lines.append(
            f"  {color_severity(len(report.regressions), '🔴', colorama.Fore.RED if has_color else '')} Regressions"
        )
        lines.append(
            f"  {color_severity(len(report.improvements), '🟢', colorama.Fore.GREEN if has_color else '')} Improvements"
        )
        lines.append(
            f"  {color_severity(len(report.warnings), '🟡', colorama.Fore.YELLOW if has_color else '')} Warnings"
        )
        lines.append(f"  ℹ️ {len(report.info_changes)} Info Changes")

        # New/Removed Operators
        if report.new_operators:
            lines.append("")
            lines.append(f"🆕 New Operators ({len(report.new_operators)})")
            for op in report.new_operators[:5]:
                lines.append(f"  - {op}")
            if len(report.new_operators) > 5:
                lines.append(f"  ... and {len(report.new_operators) - 5} more")

        if report.removed_operators:
            lines.append("")
            lines.append(f"❌ Removed Operators ({len(report.removed_operators)})")
            for op in report.removed_operators[:5]:
                lines.append(f"  - {op}")
            if len(report.removed_operators) > 5:
                lines.append(f"  ... and {len(report.removed_operators) - 5} more")

        # Show changes - REGRESSIONS
        if report.regressions:
            lines.append("")
            if has_color:
                lines.append(
                    f"{colorama.Fore.RED}🔴 Regressions ({len(report.regressions)}){colorama.Style.RESET_ALL}"
                )
            else:
                lines.append(f"🔴 Regressions ({len(report.regressions)})")
            lines.append("─" * 80)
            for change in report.regressions[:10]:
                lines.append(f"  {change.operator}: {change.test_case}")
                lines.append(
                    f"    {change.baseline_status} → {change.candidate_status}"
                )
            if len(report.regressions) > 10:
                lines.append(f"  ... and {len(report.regressions) - 10} more")

        # Show changes - IMPROVEMENTS
        if report.improvements:
            lines.append("")
            if has_color:
                lines.append(
                    f"{colorama.Fore.GREEN}🟢 Improvements ({len(report.improvements)}){colorama.Style.RESET_ALL}"
                )
            else:
                lines.append(f"🟢 Improvements ({len(report.improvements)})")
            lines.append("─" * 80)
            for change in report.improvements[:10]:
                lines.append(f"  {change.operator}: {change.test_case}")
                lines.append(
                    f"    {change.baseline_status} → {change.candidate_status}"
                )
            if len(report.improvements) > 10:
                lines.append(f"  ... and {len(report.improvements) - 10} more")

        # Show changes - WARNINGS (FIX: This section was missing proper display)
        if report.warnings:
            lines.append("")
            if has_color:
                lines.append(
                    f"{colorama.Fore.YELLOW}🟡 Warnings ({len(report.warnings)}){colorama.Style.RESET_ALL}"
                )
            else:
                lines.append(f"🟡 Warnings ({len(report.warnings)})")
            lines.append("─" * 80)
            for change in report.warnings[:10]:
                label = CHANGE_LABELS.get(change.change_type, change.change_type.value)
                lines.append(f"  {change.operator}: {change.test_case}")
                lines.append(
                    f"    {change.baseline_status} → {change.candidate_status} ({label})"
                )
            if len(report.warnings) > 10:
                lines.append(f"  ... and {len(report.warnings) - 10} more")

        # Show changes - INFO
        if report.info_changes:
            lines.append("")
            lines.append(f"ℹ️ Info Changes ({len(report.info_changes)})")
            lines.append("─" * 80)
            for change in report.info_changes[:10]:
                label = CHANGE_LABELS.get(change.change_type, change.change_type.value)
                lines.append(f"  {change.operator}: {change.test_case}")
                lines.append(
                    f"    {change.baseline_status} → {change.candidate_status} ({label})"
                )
            if len(report.info_changes) > 10:
                lines.append(f"  ... and {len(report.info_changes) - 10} more")

        # Recommendation
        lines.append("")
        lines.append("📝 Recommendation")
        lines.append("=" * 80)
        if report.summary["has_critical_changes"]:
            if has_color:
                lines.append(f"{colorama.Fore.RED}❌ REJECT{colorama.Style.RESET_ALL}")
            else:
                lines.append("❌ REJECT")
            lines.append(
                "  Critical regressions detected. Please fix before proceeding."
            )
        elif report.summary["has_warnings"]:
            if has_color:
                lines.append(
                    f"{colorama.Fore.YELLOW}⚠️ REVIEW REQUIRED{colorama.Style.RESET_ALL}"
                )
            else:
                lines.append("⚠️ REVIEW REQUIRED")
            lines.append("  Warnings detected. Please review before proceeding.")
        else:
            if has_color:
                lines.append(
                    f"{colorama.Fore.GREEN}✅ ACCEPT{colorama.Style.RESET_ALL}"
                )
            else:
                lines.append("✅ ACCEPT")
            lines.append("  No critical issues detected.")
        lines.append("=" * 80)

        return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare two test result summary files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (console output)
  python compare_test_result.py --baseline baseline/summary.json \\
      --candidate candidate/summary.json

  # Markdown output
  python compare_test_result.py --baseline baseline/summary.json \\
      --candidate candidate/summary.json --format markdown

  # Save to file
  python compare_test_result.py --baseline baseline/summary.json \\
      --candidate candidate/summary.json --output report.json

  # Custom labels
  python compare_test_result.py --baseline baseline/summary.json \\
      --candidate candidate/summary.json \\
      --baseline-label "v1.0" --candidate-label "v1.1"
        """,
    )

    parser.add_argument(
        "--baseline", required=True, type=Path, help="Path to baseline summary.json"
    )
    parser.add_argument(
        "--candidate", required=True, type=Path, help="Path to candidate summary.json"
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline",
        help="Label for baseline (default: Baseline)",
    )
    parser.add_argument(
        "--candidate-label",
        type=str,
        default="Candidate",
        help="Label for candidate (default: Candidate)",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "console"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if regressions found",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.baseline.exists():
        print(f"[ERROR] Baseline file not found: {args.baseline}")
        sys.exit(1)

    if not args.candidate.exists():
        print(f"[ERROR] Candidate file not found: {args.candidate}")
        sys.exit(1)

    # Compare
    comparator = TestComparator(
        args.baseline, args.candidate, args.baseline_label, args.candidate_label
    )

    if not comparator.load_summaries():
        print("[ERROR] Failed to load summaries")
        sys.exit(1)

    report = comparator.compare()

    # Generate output
    if args.format == "json":
        output = ReportGenerator.to_json(report)
    elif args.format == "markdown":
        output = ReportGenerator.to_markdown(report)
    else:  # console
        output = ReportGenerator.to_console(report)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"[INFO] Report written to {args.output}")
    else:
        print(output)

    # Exit with error code if regressions found
    if args.exit_code and report.summary["has_critical_changes"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
