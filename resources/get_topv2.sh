#!/usr/bin/env bash

set -euo pipefail

mkdir topv2
cd topv2

F='TOPv2_Dataset.zip'
curl -L https://fb.me/TOPv2Dataset -o $F
unzip $F
rm $F
