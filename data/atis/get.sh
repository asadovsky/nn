#!/usr/bin/env bash

set -euo pipefail

for i in {0..4}; do
  curl "http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold${i}.pkl.gz" | gunzip > "atis.fold${i}.pkl"
done
