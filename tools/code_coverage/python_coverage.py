#!/usr/bin/env python

"""
usage: python_coverage.py > python-coverage.info
"""

from os import getenv, path
from xml.etree import ElementTree


# This will generate a file use lcov data format
def process_coverage_file(xml_file):
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

    sources = root.findall("sources/source")

    source = sources[-1].text

    for cls in root.findall("packages/package/classes/class"):
        cls_filename = cls.attrib.get("filename")
        cls_filename = path.join(source, cls_filename)

        if not path.exists(cls_filename):
            continue

        print("TN:")
        print(f"SF:{cls_filename}")

        branch_index = 0
        for line in cls.findall("lines/line"):
            line_hits = line.attrib.get("hits")
            line_number = line.attrib.get("number")
            line_branch = line.attrib.get("branch")
            line_condition_coverage = line.attrib.get("condition-coverage")
            line_missing_branches = line.attrib.get("missing-branches")

            if line_branch == "true":
                line_condition_coverage = (
                    line_condition_coverage.split()[1].strip("()").split("/")
                )
                taken = int(line_condition_coverage[0])
                for _ in range(taken):
                    print(f"BRDA:{line_number},{0},{branch_index},{line_hits}")
                    branch_index += 1
                if line_missing_branches:
                    for _ in line_missing_branches.split(","):
                        print(f"BRDA:{line_number},{0},{branch_index},{0}")
                        branch_index += 1
            print(f"DA:{line_number},{line_hits}")

        print("end_of_record")


if __name__ == "__main__":
    id = getenv("PR_ID")
    sha = getenv("GITHUB_SHA")
    process_coverage_file(f"{id}-{sha}-python-coverage.xml")
