#!/usr/bin/env bash

set -euo pipefail

mkdir glove
cd glove

F='glove.6B.zip'
curl -OL http://nlp.stanford.edu/data/$F
unzip $F
rm $F
