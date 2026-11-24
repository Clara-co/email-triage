# Predicting Email Importance and Time Sensitivity — Stage 2 code

## How to run (Mac, Windows with Git Bash, or Linux)

You need Python 3.10 or newer and git.

1. Unzip this folder and open a terminal in it.
2. Install the packages:
   `pip install -r requirements.txt`
3. Download the two public datasets (about 200 MB):
   `bash setup_data.sh`
4. Run everything:
   `bash run_all.sh`

Results land in `out/tables` (CSV files with every number in the report) and `out/figs` (all figures).
Step 4 takes about 7–10 minutes; most of it is `04_models.py` tuning the Random Forest.

On Windows without Git Bash, run the two `git clone` lines from `setup_data.sh` by hand, then run the
six `python src/...` lines from `run_all.sh` one at a time.

## What each script does
| Script | Does |
| --- | --- |
| `src/01_parse.py` | Reads the raw kaminski-v mailbox into a table |
| `src/02_labels.py` | Finds his replies and labels each email High / Medium / Low |
| `src/03_features.py` | Trains the request detector on the intent sentences, builds 36 features, writes `out/features_for_knime.csv` |
| `src/04_models.py` | Tunes and tests Logistic Regression, Decision Tree, Random Forest, Naive Bayes, two baselines, ablation, feature importance, inbox ranking (NDCG), K-Means |
| `src/05_figures.py` | Draws all figures |
| `src/06_inbox_demo.py` | Draws the before/after inbox figure |

`out/` already contains the results from my run, so you can look at them before running anything.
Fonts: the figures use Inter if it is installed; otherwise matplotlib falls back to its default font.

## Data sources
- Enron Email Dataset (CMU, release of 7 May 2015), kaminski-v mailbox, de-duplicated by Akadal (2026):
  https://github.com/akadal/enron-memory-replay
- Parakweet Enron intent sentences (verified version, MIT licence):
  https://github.com/vseledkin/enron_intent_dataset_verified
