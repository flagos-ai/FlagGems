#!/usr/bin/env python

import os
import sys


def get_diff_file_lines(diff_file):
    diff_file_lines = {}
    current_file = None
    current_line = -1

    with open(diff_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("+++ "):
                current_file = line[4:]
                diff_file_lines[current_file] = []
                continue
            elif line.startswith("@@ "):
                current_line = int(line.split()[2].lstrip("+").split(",")[0])
                continue
            elif line.startswith("-"):
                continue
            elif line.startswith("+"):
                diff_file_lines[current_file].append(current_line)
            current_line += 1

    return diff_file_lines


def get_info_file_lines(info_file, diff_file):
    diff_file_lines = get_diff_file_lines(diff_file)
    current_lines = []
    current_lf = current_lh = 0
    base_path = os.environ.get("FlagGemsROOT") + "/"

    with open(info_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                current_file = line[3:]
                if current_file.startswith(base_path):
                    current_file = current_file[len(base_path) :]
                current_lines = diff_file_lines.get(current_file, [])
            elif line.startswith("DA:"):
                da = line[3:].split(",")
                if int(da[0]) in current_lines:
                    current_lf += 1
                    if not line.endswith(",0"):
                        current_lh += 1
                    print(line)
                continue
            elif line.startswith("LF:"):
                print(f"LF:{current_lf}")
                continue
            elif line.startswith("LH:"):
                print(f"LH:{current_lh}")
                continue
            print(line)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: coverage_diff.py info_file diff_file > coverage-diff.info")
        sys.exit(1)

    info_file, diff_file = sys.argv[1], sys.argv[2]

    if not (os.path.isfile(info_file) or os.path.isfile(diff_file)):
        print("Both info_file and diff_file must exist.")
        sys.exit(1)

    get_info_file_lines(info_file, diff_file)
