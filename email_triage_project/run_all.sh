#!/usr/bin/env bash
# Runs the whole pipeline in order (about 7-10 minutes).
set -e
cd "$(dirname "$0")"
python src/01_parse.py
python src/02_labels.py
python src/03_features.py
python src/04_models.py
python src/05_figures.py
python src/06_inbox_demo.py
echo "Done. Results are in out/tables and out/figs."
