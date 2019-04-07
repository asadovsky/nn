#!/usr/bin/env bash

set -euo pipefail

F='glove.6B.zip'
curl -O https://nlp.stanford.edu/data/$F
unzip $F
rm $F
