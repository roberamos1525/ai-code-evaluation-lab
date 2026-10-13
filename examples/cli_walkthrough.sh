#!/usr/bin/env sh
# End-to-end CLI tour. Run from the project root with the package installed:
#   sh examples/cli_walkthrough.sh [state-dir]
# Uses a throw-away state directory so it never touches var/.
set -e
HOME_DIR="${1:-./var/walkthrough}"
CE="codeeval --home $HOME_DIR"

echo "== datasets"
$CE datasets list
$CE datasets validate core-v1 --execute | tail -2

echo "== import a HumanEval-style file and evaluate it"
$CE datasets import examples/datasets/humaneval_style.jsonl "$HOME_DIR/datasets/he-example.jsonl" --name he-example
$CE run --dataset "$HOME_DIR/datasets/he-example.jsonl" --adapter oracle --quiet

echo "== two adapters on the built-in dataset (5 samples each -> pass@1 and pass@5)"
$CE run --adapter oracle --samples 5 --k 1,5 --baseline main --quiet
$CE run --adapter "mutation:bug_rate=0.5,seed=7" --samples 5 --k 1,5 --quiet --report-dir "$HOME_DIR/reports"

echo "== compare, report, gate"
$CE compare baseline:main latest --format md --out "$HOME_DIR/reports/comparison.md"
$CE report latest --format all --out-dir "$HOME_DIR/reports"
$CE regression --baseline baseline:main --candidate latest || echo "(exit $?: regression detected, as expected)"

echo "== single submissions and the policy checker"
$CE evaluate examples/solutions/rle_encode.py --task core/rle-encode
$CE evaluate examples/solutions/rle_encode_buggy.py --task core/rle-encode || true
$CE check examples/solutions/unsafe.py --strictness strict || true
