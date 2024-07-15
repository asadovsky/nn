#!/usr/bin/env bash

set -euo pipefail

mkdir hellaswag
cd hellaswag

F=(
  'hellaswag_train.jsonl'
  'hellaswag_val.jsonl'
  'hellaswag_test.jsonl'
)

for f in "${F[@]}"; do
  curl -OL "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/${f}"
done
