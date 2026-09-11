#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Send Test status notification to Feishu via Bot API."""

import json
import os
import sys
import urllib.request

# Feishu credentials — skip gracefully if not configured
FEISHU_VARS = ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_CHAT_ID"]

# Required environment variables (script exits with error if any are missing)
REQUIRED_VARS = [
    # CI status (computed by the calling workflow)
    "CI_STATUS",
    # GitHub default env vars (available automatically on every runner)
    "GITHUB_WORKFLOW",
    "GITHUB_REPOSITORY",
    "GITHUB_REF_NAME",
    "GITHUB_SHA",
    "GITHUB_ACTOR",
    "GITHUB_RUN_ID",
    "GITHUB_SERVER_URL",
    "GITHUB_EVENT_NAME",
]

# Optional environment variables (logged as warning if missing)
OPTIONAL_VARS = [
    "PLATFORM",
    # Override vars for workflow_run context — when set, these take precedence
    # over the GitHub default env vars (which reflect the notification workflow,
    # not the original CI run).
    "OVERRIDE_REF",  # e.g. github.event.workflow_run.head_branch
    "OVERRIDE_SHA",  # e.g. github.event.workflow_run.head_sha
    "OVERRIDE_RUN_URL",  # e.g. github.event.workflow_run.html_url
    "OVERRIDE_WORKFLOW",  # e.g. github.event.workflow_run.name
    "PR_NUMBER",  # e.g. github.event.workflow_run.pull_requests[0].number
    "PR_TITLE",  # PR title from GitHub API
]


def check_env_vars() -> bool:
    """Validate environment variables. Return True if valid, False to skip."""
    # Check Feishu credentials first — missing means "not configured", not an error
    missing_feishu = [v for v in FEISHU_VARS if not os.environ.get(v, "").strip()]
    if missing_feishu:
        print(
            f"::notice::Feishu notification skipped — "
            f"missing credentials: {', '.join(missing_feishu)}"
        )
        return False

    # Check remaining required vars — these should always be present
    invalid = [v for v in REQUIRED_VARS if not os.environ.get(v, "").strip()]

    for v in OPTIONAL_VARS:
        if not os.environ.get(v, "").strip():
            print(f"::warning::Optional environment variable '{v}' is not set")

    if invalid:
        for v in invalid:
            print(
                f"::warning::Required environment variable '{v}' is missing or empty",
                file=sys.stderr,
            )
        return False

    return True


def main():
    if not check_env_vars():
        # Exit 0 — missing config is not a workflow failure
        return

    app_id = os.environ["FEISHU_APP_ID"]
    app_secret = os.environ["FEISHU_APP_SECRET"]
    chat_id = os.environ["FEISHU_CHAT_ID"]
    ci_status = os.environ["CI_STATUS"].strip()
    platform = os.environ.get("PLATFORM", "")

    # GitHub default environment variables (always available on runners)
    # Override vars take precedence — used when called from workflow_run context
    workflow = os.environ.get("OVERRIDE_WORKFLOW") or os.environ["GITHUB_WORKFLOW"]
    repo = os.environ["GITHUB_REPOSITORY"]
    ref = os.environ.get("OVERRIDE_REF") or os.environ["GITHUB_REF_NAME"]
    sha = os.environ.get("OVERRIDE_SHA") or os.environ["GITHUB_SHA"]
    actor = os.environ["GITHUB_ACTOR"]
    run_id = os.environ["GITHUB_RUN_ID"]
    server_url = os.environ["GITHUB_SERVER_URL"]
    event_name = os.environ["GITHUB_EVENT_NAME"]
    pr_number = os.environ.get("PR_NUMBER", "")
    pr_title = os.environ.get("PR_TITLE", "")

    short_sha = sha[:7]
    run_url = (
        os.environ.get("OVERRIDE_RUN_URL")
        or f"{server_url}/{repo}/actions/runs/{run_id}"
    )

    # --- 1. Obtain tenant_access_token ---
    token_req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(token_req) as resp:
        token_data = json.loads(resp.read())

    tenant_token = token_data.get("tenant_access_token", "")
    if not tenant_token:
        print(
            f"::warning::Failed to obtain tenant_access_token: {token_data}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 2. Build message card ---
    status_map = {
        "success": ("\u2705", "Success", "green"),
        "cancelled": ("\u26a0\ufe0f", "Cancelled", "orange"),
    }
    emoji, text, color = status_map.get(ci_status, ("\u274c", "Failed", "red"))

    title = f"Test {text} \u2014 {repo}"
    if platform:
        title = f"Test {text} [{platform}] \u2014 {repo}"

    card_content = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Workflow:** {workflow}",
                        },
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Status:** {emoji} {text}",
                        },
                    },
                    {
                        "is_short": True,
                        "text": {"tag": "lark_md", "content": f"**Branch:** {ref}"},
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Trigger:** {event_name} by {actor}",
                        },
                    },
                    {
                        "is_short": False,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Commit:** {short_sha}",
                        },
                    },
                ]
                + (
                    [
                        {
                            "is_short": False,
                            "text": {
                                "tag": "lark_md",
                                "content": f"**PR:** [{repo}#{pr_number}]({server_url}/{repo}/pull/{pr_number})",
                            },
                        },
                    ]
                    if pr_number
                    else []
                )
                + (
                    [
                        {
                            "is_short": False,
                            "text": {
                                "tag": "lark_md",
                                "content": f"**Title:** {pr_title}",
                            },
                        },
                    ]
                    if pr_title
                    else []
                ),
            },
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "View Run"},
                        "url": run_url,
                        "type": "primary",
                    }
                ],
            },
        ],
    }

    body = json.dumps(
        {
            "receive_id": chat_id,
            "msg_type": "interactive",
            "content": json.dumps(card_content),
        }
    ).encode()

    # --- 3. Send message ---
    msg_req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        data=body,
        headers={
            "Authorization": f"Bearer {tenant_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(msg_req) as resp:
        result = json.loads(resp.read())

    if result.get("code", -1) != 0:
        print(f"::warning::Feishu API returned error: {result}", file=sys.stderr)
        sys.exit(1)

    print("Feishu notification sent successfully.")


if __name__ == "__main__":
    main()
