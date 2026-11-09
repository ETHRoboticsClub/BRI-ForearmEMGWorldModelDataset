#!/usr/bin/env bash
# Bring the result of a `sky exec` build back: dvc.lock from the machine, then the built data from S3.
# Usage: sky/fetch.sh [cluster]    (default: femg-build)
set -euo pipefail
cluster="${1:-femg-build}"
cd "$(dirname "$0")/.."

rsync -av "$cluster:sky_workdir/dvc.lock" .
dvc pull
echo
echo "Commit the build with: git add dvc.lock && git commit -m '...' && git push"
