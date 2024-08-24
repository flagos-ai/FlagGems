#!/usr/bin/env python

"""
usage: pull_request.py [files|diff] pull_id
"""

import argparse
import os

from github import Github

# Retrieve the GitHub API token from environment variables
token = os.getenv("GITHUB_API_TOKEN")


def get_pull(pull_id):
    github = Github(token, timeout=60)
    repo = github.get_repo("FlagOpen/FlagGems")
    return repo.get_pull(pull_id)


def get_files(args):
    pull = get_pull(args.pull_id)
    flag_gems_root = os.environ.get("FlagGemsROOT")
    for file in pull.get_files():
        print(f"{flag_gems_root}/{file.filename}")


def show_diff(args):
    pull = get_pull(args.pull_id)
    for file in pull.get_files():
        print(f"+++ {file.filename}")
        print(file.patch)


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Fetch pull request details from GitHub."
    )
    subparsers = parser.add_subparsers()

    # Subparser for 'files' command
    files_parser = subparsers.add_parser("files", help="List files in the pull request")
    files_parser.add_argument("pull_id", type=int, help="ID of the pull request")
    files_parser.set_defaults(func=get_files)

    # Subparser for 'diff' command
    diff_parser = subparsers.add_parser("diff", help="Show diff of the pull request")
    diff_parser.add_argument("pull_id", type=int, help="ID of the pull request")
    diff_parser.set_defaults(func=show_diff)

    # Parse and execute the command
    args = parser.parse_args()
    args.func(args)
