#!/usr/bin/env bash
# Downloads the two public datasets into ./data (needs git; about 200 MB).
set -e
cd "$(dirname "$0")"
mkdir -p data out/figs out/tables
[ -d data/emr ]    || git clone --depth 1 https://github.com/akadal/enron-memory-replay data/emr
[ -d data/intent ] || git clone --depth 1 https://github.com/vseledkin/enron_intent_dataset_verified data/intent
echo "Data ready."
