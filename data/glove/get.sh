#!/usr/bin/env bash

set -euo pipefail

F='glove.6B.zip'
curl -L -O http://nlp.stanford.edu/data/$F
unzip $F
rm $F
