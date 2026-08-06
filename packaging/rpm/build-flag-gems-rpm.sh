#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BASE_IMAGE_VERSION="${1:-43}"
IMAGE_TAG="flag-gems-rpm:f${BASE_IMAGE_VERSION}"
OUTPUT_DIR="${PROJECT_DIR}/rpm-packages"

docker build --network=host \
    -f "${SCRIPT_DIR}/dockerfiles/Dockerfile.rpm" \
    --build-arg BASE_IMAGE_VERSION="$BASE_IMAGE_VERSION" \
    -t "$IMAGE_TAG" "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
CID=$(docker create "$IMAGE_TAG")
docker cp "$CID:/output/." "$OUTPUT_DIR/"
docker rm "$CID" > /dev/null

echo ""; echo ">>> Output:"; ls -lh "$OUTPUT_DIR"
