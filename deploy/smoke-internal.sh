#!/usr/bin/env bash
# Internal smoke test on server (Phase 2 step 6). App must be on 127.0.0.1:8000.
set -euo pipefail

BASE_URL="${DRAKONSUB_SMOKE_URL:-http://127.0.0.1:8000}"

echo "Smoke: GET ${BASE_URL}/api/health"
health_json="$(curl -fsS "${BASE_URL}/api/health")"
echo "${health_json}"
echo "${health_json}" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('status')=='ok', d; assert d.get('jobs_root_writable') is True, d"

echo "Smoke: GET ${BASE_URL}/api/defaults"
curl -fsS "${BASE_URL}/api/defaults" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'translation_engine' in d, d"

echo "OK: internal smoke passed"
