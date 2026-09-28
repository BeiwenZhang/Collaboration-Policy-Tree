#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export BUILD_ENV="${BUILD_ENV:-production}"
docker compose up --build -d
echo "Human-AI demo: http://localhost:${HOST_PORT:-8080}"

