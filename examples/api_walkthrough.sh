#!/usr/bin/env sh
# Exercise the REST API with curl. Start the server first, e.g.
#   CODEEVAL_API_KEYS=change-me-0123456789 codeeval serve --port 8000
# then: BASE=http://127.0.0.1:8000 KEY=change-me-0123456789 sh examples/api_walkthrough.sh
set -e
BASE="${BASE:-http://127.0.0.1:8000}"
AUTH="X-API-Key: ${KEY:-change-me-0123456789}"
JSON="Content-Type: application/json"

curl -s "$BASE/healthz"; echo
curl -s -H "$AUTH" "$BASE/v1/datasets/core-v1"; echo

RUN=$(curl -s -H "$AUTH" -H "$JSON" -X POST "$BASE/v1/runs" \
  -d '{"adapter": "mutation:bug_rate=0.4,seed=1", "samples": 3, "ks": [1, 3], "limit": 12}')
echo "$RUN"
RUN_ID=$(echo "$RUN" | python -c "import sys, json; print(json.load(sys.stdin)['run_id'])")

while :; do
  STATUS=$(curl -s -H "$AUTH" "$BASE/v1/runs/$RUN_ID" | python -c "import sys, json; print(json.load(sys.stdin)['status'])")
  [ "$STATUS" = "pending" ] || [ "$STATUS" = "running" ] || break
  sleep 0.5
done
echo "status: $STATUS"

curl -s -H "$AUTH" "$BASE/v1/runs/$RUN_ID/failures"; echo
curl -s -H "$AUTH" "$BASE/v1/runs/$RUN_ID/report?format=md" | head -30

curl -s -H "$AUTH" -H "$JSON" -X POST "$BASE/v1/datasets/core-v1/tasks/core/rle-encode/submissions" \
  -d '{"code": "def rle_encode(text):\n    return \"\"\n"}'; echo
curl -s -H "$AUTH" -H "$JSON" -X POST "$BASE/v1/policy/check" \
  -d '{"code": "import socket", "strictness": "strict"}'; echo
