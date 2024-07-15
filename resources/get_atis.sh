#!/usr/bin/env bash

set -euo pipefail

mkdir atis
cd atis

F=(
  'atis-2.dev.iob'
  'atis-2.dev.w-intent.iob'
  'atis-2.train.iob'
  'atis-2.train.w-intent.iob'
  'atis.test.iob'
  'atis.test.w-intent.iob'
  'atis.train.iob'
  'atis.train.w-intent.iob'
)

for f in "${F[@]}"; do
  curl -OL "https://raw.githubusercontent.com/yvchen/JointSLU/master/data/${f}"
done
