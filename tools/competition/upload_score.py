#!/usr/bin/env python3

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Allowed dimension keys and their max scores for score_details
SCORE_DIMENSIONS = {
    "functional_correctness": 30,  # Functional correctness
    "performance": 20,  # Performance competitiveness
    "test_coverage": 20,  # Test case completeness
    "adaptability": 10,  # Open-source adaptability
    "compatibility": 10,  # Cross-platform compatibility
    "readability": 10,  # Code readability
}


def generate_signature(params: dict, api_secret: str) -> str:
    """Generate request signature (same algorithm as evaluation_client.py).

    Rules: only sign scalar fields, skip dict/list fields.
    Signature covers identity and operation fields; nested detail data is secured by API Key + HTTPS.
    """
    sorted_keys = sorted(params.keys())
    parts = []
    for k in sorted_keys:
        v = params[k]
        if isinstance(v, (dict, list)):
            continue
        parts.append(f"{k}={v}")
    sign_string = "&".join(parts)
    sign_string = f"{sign_string}&api_secret={api_secret}"
    return hashlib.sha256(sign_string.encode()).hexdigest()


def validate_score_details(score_details: dict) -> str:
    """Validate score_details dimensions, return error message (empty string means pass)."""
    for key, value in score_details.items():
        if key not in SCORE_DIMENSIONS:
            return f"Unknown dimension '{key}', allowed dimensions: {', '.join(SCORE_DIMENSIONS.keys())}"
        if not isinstance(value, (int, float)):
            return f"Dimension '{key}' value must be a number, got: {value}"
        if value < 0 or value > SCORE_DIMENSIONS[key]:
            return f"Dimension '{key}' value out of range [0, {SCORE_DIMENSIONS[key]}], got: {value}"
    return ""


def upload_score(
    api_base_url: str,
    api_key: str,
    api_secret: str,
    name: str,
    github_id: str,
    github_pr: str,
    score: float,
    note: str = None,
    score_details: dict = None,
) -> bool:
    """
    Upload score to FlagOS platform.

    Args:
        api_base_url: FlagOS API base URL
        api_key: Evaluation API Key
        api_secret: Evaluation API Secret
        name: Operator name (must exactly match the registered name on the platform)
        github_id: Contestant's GitHub ID
        github_pr: GitHub PR link
        score: Total score (0-100)
        note: Optional note
        score_details: Optional per-dimension scores, format:
            {"functional_correctness": 27, "performance": 18, ...}

    Returns:
        bool: Whether the upload was successful
    """
    url = api_base_url.rstrip("/") + "/api/v1/evaluation/track1-score"
    timestamp = str(int(time.time()))

    # Build request body
    data = {
        "name": name,
        "github_id": github_id,
        "github_pr": github_pr,
        "score": score,
    }
    if note:
        data["note"] = note
    if score_details:
        data["score_details"] = score_details

    # Compute signature (score_details is a dict, skipped by the signing function)
    sign_params = {**data, "timestamp": timestamp}
    signature = generate_signature(sign_params, api_secret)

    headers = {
        "Content-Type": "application/json",
        "x-eval-api-key": api_key,
        "x-eval-timestamp": timestamp,
        "x-eval-signature": signature,
    }

    try:
        details_str = f", score_details={score_details}" if score_details else ""
        logger.info(
            f"Uploading score: name={name}, github_id={github_id}, score={score}{details_str}"
        )
        response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()

        if response.status_code == 200 and result.get("code") == 200:
            logger.info(f"Upload successful: {result.get('message')}")
            return True
        else:
            logger.error(f"Upload failed: {result}")
            return False

    except requests.RequestException as e:
        logger.error(f"Request exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Competition Track 1 CI/CD score upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Score dimensions and max scores:
  functional_correctness  Functional correctness   max 30
  performance             Performance              max 20
  test_coverage           Test case completeness   max 20
  adaptability            Open-source adaptability max 10
  compatibility           Cross-platform compat.   max 10
  readability             Code readability         max 10

Example:
  --score-details '{"functional_correctness": 27, "performance": 18,
                    "test_coverage": 16, "adaptability": 9,
                    "compatibility": 8, "readability": 10}'
""",
    )
    parser.add_argument("--api-url", help="FlagOS API URL", default=None)
    parser.add_argument("--api-key", help="Evaluation API Key", default=None)
    parser.add_argument("--api-secret", help="Evaluation API Secret", default=None)
    parser.add_argument(
        "--name",
        required=True,
        help="Operator name (must exactly match the registered name, case-sensitive)",
    )
    parser.add_argument(
        "--github-id", required=True, help="Contestant's GitHub username"
    )
    parser.add_argument("--github-pr", required=True, help="GitHub PR link")
    parser.add_argument(
        "--score", required=True, type=float, help="Total score (0-100)"
    )
    parser.add_argument("--note", default=None, help="Optional note")
    parser.add_argument(
        "--score-details",
        default=None,
        help='Per-dimension score JSON, e.g. \'{"functional_correctness": 27, "performance": 18, ...}\'',
    )

    args = parser.parse_args()

    # Prefer CLI args, fall back to environment variables
    api_url = args.api_url or os.environ.get("EVAL_API_BASE_URL", "")
    api_key = args.api_key or os.environ.get("EVAL_API_KEY", "")
    api_secret = args.api_secret or os.environ.get("EVAL_API_SECRET", "")

    if not all([api_url, api_key, api_secret]):
        logger.error(
            "Missing required parameters: api-url, api-key, api-secret (provide via CLI or environment variables)"
        )
        sys.exit(1)

    # Parse score_details
    score_details = None
    if args.score_details:
        try:
            score_details = json.loads(args.score_details)
        except json.JSONDecodeError as e:
            logger.error(f"--score-details JSON format error: {e}")
            sys.exit(1)

        error = validate_score_details(score_details)
        if error:
            logger.error(f"--score-details validation failed: {error}")
            sys.exit(1)

    success = upload_score(
        api_base_url=api_url,
        api_key=api_key,
        api_secret=api_secret,
        name=args.name,
        github_id=args.github_id,
        github_pr=args.github_pr,
        score=args.score,
        note=args.note,
        score_details=score_details,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
