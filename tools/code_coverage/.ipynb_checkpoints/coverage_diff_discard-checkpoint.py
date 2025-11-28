#!/usr/bin/env python

import os
import re
import sys


def get_discard_file_lines(discard_file):
    flag_gems_root = os.environ.get("FlagGemsROOT")
    dicard_file_lines = {}
    with open(discard_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith(flag_gems_root + "/"):
                current_file = line[len(flag_gems_root) + 1 :]
                dicard_file_lines[current_file] = []
                continue

            elif line.startswith("--- "):
                pattern = r"(\d+) : (\d+)"
                match = re.search(pattern, line)
                if match:
                    start, end = map(int, match.groups())
                # Note: Use (start + 1) instead of start
                # Because we take the definition of the JIT function into account
                for i in range(start + 1, end + 1):
                    dicard_file_lines[current_file].append(i)
    return dicard_file_lines


def get_info_file_lines(info_file, discard_file):
    discard_file_lines = get_discard_file_lines(discard_file)
    discard_lines = []
    num_rm_lines = 0
    base_path = os.environ.get("FlagGemsROOT") + "/"

    with open(info_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                num_rm_lines = 0
                current_file = line[3:]
                if current_file.startswith(base_path):
                    current_file = current_file[len(base_path) :]
                discard_lines = discard_file_lines.get(current_file, [])
            elif line.startswith("DA:"):
                da = line[3:].split(",")
                if int(da[0]) in discard_lines:
                    num_rm_lines -= 1
                    continue
                else:
                    print(line)
                    continue
            elif line.startswith("LF:"):
                lf = line.split(":")
                print(f"LF:{int(lf[1]) + num_rm_lines}")
                continue
            elif line.startswith("LH:"):
                lh = line.split(":")
                print(f"LH:{int(lh[1]) + num_rm_lines}")
                continue
            print(line)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: coverage_diff.py info_file discard_file > python-coverage-discard-diff.info"
        )
        sys.exit(1)

    info_file, discard_file = sys.argv[1], sys.argv[2]

    if not (os.path.isfile(info_file) or os.path.isfile(discard_file)):
        print("Both info_file and discard_file must exist.")
        sys.exit(1)

    get_info_file_lines(info_file, discard_file)
