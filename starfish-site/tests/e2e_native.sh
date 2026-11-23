#!/usr/bin/env bash
# End-to-end test of starfish-site, three native sites on this machine.
#
#   starfish-site/tests/e2e_native.sh STORES [ROUNDS] [EPOCHS]
#
# STORES holds site-a, site-b, site-c and eval-a as make babelbrain-demo-stores
# writes them, readable by this user. The demo writes them as root, so copy
# them first. starfish-site must be on PATH, or STARFISH_SITE must name the
# command. A test router from the starfish-router image runs in Docker as
# compose project starfish-site-e2e on 127.0.0.1:18100 and is removed at the
# end, with the sites' folders. Site A is the coordinator with eval-a as its
# test set, and site C trains on the CPU through BABELBRAIN_FL_DEVICE.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORES="$(cd "${1:?usage: e2e_native.sh STORES [ROUNDS] [EPOCHS]}" && pwd)"
ROUNDS="${2:-3}"
EPOCHS="${3:-4}"
SITE="${STARFISH_SITE:-starfish-site}"
API=http://127.0.0.1:18100/starfish/api/v1
WORK="$(mktemp -d)"
COMPOSE=(docker compose -f "$HERE/e2e_router.yaml")

cleanup() {
  for x in a b c; do
    [ -d "$WORK/site-$x" ] && "$SITE" --home "$WORK/site-$x" stop --now --wait 15 >/dev/null 2>&1 || true
  done
  "${COMPOSE[@]}" down -v >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

"${COMPOSE[@]}" up -d --wait
code() {
  curl -s -u admin:1234 -H 'Content-Type: application/json' -d '{"note": "e2e"}' "$API/enrolment-codes/" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["code"])'
}
"$SITE" --home "$WORK/site-a" enrol --router "$API" --code "$(code)" --name "e2e A" \
  --store "$STORES/site-a" --eval-store "$STORES/eval-a"
"$SITE" --home "$WORK/site-b" enrol --router "$API" --code "$(code)" --name "e2e B" --store "$STORES/site-b"
"$SITE" --home "$WORK/site-c" enrol --router "$API" --code "$(code)" --name "e2e C" --store "$STORES/site-c" \
  --device cpu
"$SITE" --home "$WORK/site-a" check --quick
for x in a b c; do
  "$SITE" --home "$WORK/site-$x" start > "$WORK/site-$x.log" 2>&1 < /dev/null &
done
python3 "$HERE/e2e_run.py" "$API" "$ROUNDS" "$EPOCHS" "e2e A" "e2e B" "e2e C"
