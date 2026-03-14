#!/bin/bash
#
# Generate FlagGems operators documentation
#
# Usage:
#   ./tools/gen_operators.sh              # Generate docs/operators.md
#   ./tools/gen_operators.sh -o custom.md # Specify output path
#   ./tools/gen_operators.sh --check       # Check if docs need update

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Parse arguments
OUTPUT_FILE="docs/operators.md"
CHECK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --check)
            CHECK_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$CHECK_MODE" = true ]; then
    # Generate to temp file and compare
    TEMP_FILE=$(mktemp)
    python3 -m tools.generate_operators_doc -o "$TEMP_FILE"

    if diff -q "$OUTPUT_FILE" "$TEMP_FILE" > /dev/null 2>&1; then
        echo "operators.md is up to date"
        rm "$TEMP_FILE"
        exit 0
    else
        echo "operators.md is out of date"
        echo "Run './tools/gen_operators.sh' to update"
        rm "$TEMP_FILE"
        exit 1
    fi
else
    python3 -m tools.generate_operators_doc -o "$OUTPUT_FILE"
    echo "Generated: $OUTPUT_FILE"
fi
