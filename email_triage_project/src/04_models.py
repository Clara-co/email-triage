"""Stage 2 · Step 4 — Modelling: LR, Decision Tree, Random Forest, Naive Bayes (+ baselines, K-Means).

Protocol: chronological hold-out (train = Dec 1999-Dec 2000, test = Jan-Apr 2001);
hyper-parameters tuned with 4-fold TimeSeriesSplit inside the training period, scoring = macro-F1.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support, confusion_matrix, roc_auc_score,
                             average_precision_score, ndcg_score, silhouette_score)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
T = f"{OUT}/tables"
CLASSES = ["High", "Medium", "Low"]
G = json.load(open(f"{OUT}/feature_groups.json"))
NUM = sum(G.values(), [])

df = pd.read_parquet(f"{OUT}/features.parquet")
df = df[df.in_window].sort_values("utc").reset_index(drop=True)
for c in ["priority", "text", "sender", "subject"]:
    df[c] = df[c].astype(object)
tr, te = df[df.split == "train"], df[df.split == "test"]
ytr, yte = tr.priority.to_numpy(dtype=object), te.priority.to_numpy(dtype=object)
tscv = TimeSeriesSplit(4)


def text_block(k):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                                               stop_words="english", max_features=30000)),
                     ("chi2", SelectKBest(chi2, k=k))])


def make(model, num_cols=NUM, text=True, k=1000):
    if model == "nb":
        num_t = Pipeline([("bin", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile"))])
    elif model == "lr":
        num_t = StandardScaler()
    else:
        num_t = "passthrough"
    parts = []
    if num_cols:
        parts.append(("num", num_t, list(num_cols)))
    if text:
        parts.append(("txt", text_block(k), "text"))
    pre = ColumnTransformer(parts)
    est = {"lr": LogisticRegression(max_iter=4000, class_weight="balanced"),
           "dt": DecisionTreeClassifier(class_weight="balanced", random_state=0),
           "rf": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=0),
           "nb": ComplementNB()}[model]
    return Pipeline([("pre", pre), ("clf", est)])


GRIDS = {
    "lr": {"clf__C": [0.03, 0.1, 0.3, 1.0], "pre__txt__chi2__k": [300, 1000, 3000]},
    "dt": {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [5, 20, 50], "pre__txt__chi2__k": [100, 300]},
    "rf": {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 3, 8], "pre__txt__chi2__k": [300, 1000]},
    "nb": {"clf__alpha": [0.1, 0.3, 1.0, 3.0], "pre__txt__chi2__k": [300, 1000, 3000]},
}
NAMES = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest", "nb": "Naive Bayes (Complement)"}


def evaluate(name, yhat, proba=None):
    p, r, f, s = precision_recall_fscore_support(yte, yhat, labels=CLASSES, zero_division=0)
    row = dict(model=name, accuracy=accuracy_score(yte, yhat), macro_f1=f1_score(yte, yhat, average="macro"),
               weighted_f1=f1_score(yte, yhat, average="weighted"),
               balanced_acc=balanced_accuracy_score(yte, yhat), kappa=cohen_kappa_score(yte, yhat))
    for c, pi, ri, fi in zip(CLASSES, p, r, f):
        row.update({f"{c}_precision": pi, f"{c}_recall": ri, f"{c}_f1": fi})
    if proba is not None:
        Y = np.column_stack([(yte == c).astype(int) for c in CLASSES])
        row["macro_auc"] = roc_auc_score(Y, proba, average="macro")
        row["High_pr_auc"] = average_precision_score(Y[:, 0], proba[:, 0])
        row["respond_auc"] = roc_auc_score(1 - Y[:, 2], proba[:, 0] + proba[:, 1])  # any response vs none
    # bootstrap 95% CI for macro-F1
    rng = np.random.default_rng(0)
    bs = [f1_score(yte[i], np.asarray(yhat)[i], average="macro")
          for i in (rng.integers(0, len(yte), len(yte)) for _ in range(1000))]
    row["macro_f1_lo"], row["macro_f1_hi"] = np.percentile(bs, [2.5, 97.5])
    return row


results, fitted, probas, preds, best_params = [], {}, {}, {}, {}

# --- baselines
MAJ = pd.Series(ytr).mode()[0]
results.append(evaluate(f"Baseline: always '{MAJ}'", np.array([MAJ] * len(te))))
# hand-written triage rules (no learning): personal + asking -> High; personal or in an active thread -> Medium
personal = (te.sole_recipient == 1) | ((te.owner_in_to == 1) & (te.n_recip <= 3))
asking = (te.intent_max > 0.5) | (te.n_question > 0)
noise = (te.bulk_flag == 1) | (te.n_recip >= 10) | (te.sender_automated == 1) | (te.unsubscribe == 1)
rule = np.where(noise, "Low", np.where(personal & asking, "High",
                np.where(personal | (te.owner_in_thread_before == 1), "Medium", "Low")))
results.append(evaluate("Baseline: hand-written rules", rule))
preds["rules"] = rule

for key in ["lr", "dt", "rf", "nb"]:
    t0 = time.time()
    gs = GridSearchCV(make(key), GRIDS[key], cv=tscv, scoring="f1_macro", n_jobs=1 if key == "rf" else -1)
    fit_kw = {}
    if key == "nb":  # ComplementNB has no class_weight: use equivalent balanced sample weights
        from sklearn.utils.class_weight import compute_sample_weight
        fit_kw["clf__sample_weight"] = compute_sample_weight("balanced", ytr)
    gs.fit(tr, ytr, **fit_kw)
    m = gs.best_estimator_
    order = [list(m.classes_).index(c) for c in CLASSES]
    pr = m.predict_proba(te)[:, order]
    yhat = m.predict(te)
    fitted[key], probas[key], preds[key] = m, pr, yhat
    best_params[NAMES[key]] = {k: (v if not isinstance(v, (np.integer,)) else int(v)) for k, v in gs.best_params_.items()}
    row = evaluate(NAMES[key], yhat, pr)
    row["cv_macro_f1"] = gs.best_score_
    row["fit_seconds"] = time.time() - t0
    results.append(row)
    print(f"{NAMES[key]:28s} test macro-F1={row['macro_f1']:.3f}  acc={row['accuracy']:.3f}  "
          f"High-recall={row['High_recall']:.3f}  cv={gs.best_score_:.3f}  {gs.best_params_}")

res = pd.DataFrame(results)
res.to_csv(f"{T}/model_comparison.csv", index=False)
json.dump(best_params, open(f"{T}/best_params.json", "w"), indent=2)
for k in fitted:
    cm = confusion_matrix(yte, preds[k], labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(f"{T}/cm_{k}.csv")
np.save(f"{OUT}/test_probas.npy", probas)
joblib.dump(fitted, f"{OUT}/models.joblib")

best = res.iloc[2:].sort_values("macro_f1", ascending=False).iloc[0]
best_key = {v: k for k, v in NAMES.items()}[best.model]
print("best:", best.model)

# --- feature-group ablation with every model's tuned params
abl = []
configs = {"Content (text + content features)": (G["content"], True),
           "Metadata only": (G["metadata"], False),
           "Behavioral only": (G["behavioral"], False),
           "Metadata + Behavioral": (G["metadata"] + G["behavioral"], False),
           "Text only (TF-IDF)": ([], True),
           "All features": (NUM, True)}
for key in ["lr", "rf"]:
    params = {k: v for k, v in fitted[key].get_params().items() if k.startswith("clf__")}
    k_txt = fitted[key].get_params()["pre__txt__chi2__k"]
    for cname, (cols, txt) in configs.items():
        m = make(key, cols, txt, k=k_txt).set_params(**params).fit(tr, ytr)
        yhat = m.predict(te)
        abl.append(dict(model=NAMES[key], features=cname, macro_f1=f1_score(yte, yhat, average="macro"),
                        High_recall=precision_recall_fscore_support(yte, yhat, labels=["High"], zero_division=0)[1][0]))
pd.DataFrame(abl).to_csv(f"{T}/ablation.csv", index=False)
print(pd.DataFrame(abl).pivot(index="features", columns="model", values="macro_f1").round(3))

# --- interpretability
# (a) permutation importance of the numeric features + the text block, on the test set (RF)
rf = fitted["rf"]
X_perm = te.copy()
imp_rows = []
rng = np.random.default_rng(0)
base = f1_score(yte, rf.predict(te), average="macro")
for col in NUM + ["text"]:
    drops = []
    for rep in range(5):
        Xp = X_perm.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        drops.append(base - f1_score(yte, rf.predict(Xp), average="macro"))
    grp = "text (TF-IDF)" if col == "text" else next(g for g, cs in G.items() if col in cs)
    imp_rows.append(dict(feature=col, group=grp, importance=np.mean(drops), sd=np.std(drops)))
pd.DataFrame(imp_rows).sort_values("importance", ascending=False).to_csv(f"{T}/perm_importance_rf.csv", index=False)

# (b) LR coefficients for the High class (numeric + words)
lr = fitted["lr"]
pre = lr.named_steps["pre"]
vocab = pre.named_transformers_["txt"].named_steps["tfidf"].get_feature_names_out()
sel = pre.named_transformers_["txt"].named_steps["chi2"].get_support()
names = NUM + list(vocab[sel])
coef = pd.DataFrame(lr.named_steps["clf"].coef_.T, index=names, columns=lr.named_steps["clf"].classes_)
coef["kind"] = ["feature"] * len(NUM) + ["word"] * int(sel.sum())
coef.to_csv(f"{T}/lr_coefficients.csv")

# --- triage simulation: rank each test day's mail by expected priority
P = probas[best_key]
score = 2 * P[:, 0] + 1 * P[:, 1]
gain = te.priority.map({"High": 2, "Medium": 1, "Low": 0}).values
days = te.utc.dt.tz_convert("US/Central").dt.date.values
tri = []
for d in np.unique(days):
    i = np.where(days == d)[0]
    if len(i) < 5 or gain[i].max() == 0:
        continue
    chron = -te.utc.values[i].astype("int64").astype(float)  # negated arrival time; -chron ranks newest first
    tri.append(dict(day=str(d), n=len(i),
                    ndcg_model=ndcg_score([gain[i]], [score[i]], k=10),
                    ndcg_chrono=ndcg_score([gain[i]], [-chron], k=10),
                    ndcg_random=np.mean([ndcg_score([gain[i]], [rng.random(len(i))], k=10) for _ in range(20)]),
                    high_in_top5_model=int((te.priority.values[i][np.argsort(-score[i])[:5]] == "High").sum()),
                    high_in_top5_chrono=int((te.priority.values[i][np.argsort(-te.utc.values[i].astype("int64"))[:5]] == "High").sum()),
                    n_high=int((te.priority.values[i] == "High").sum())))
tri = pd.DataFrame(tri)
tri.to_csv(f"{T}/triage_days.csv", index=False)
print("triage NDCG@10  model=%.3f  newest-first=%.3f  random=%.3f  (days=%d)" %
      (tri.ndcg_model.mean(), tri.ndcg_chrono.mean(), tri.ndcg_random.mean(), len(tri)))

# cumulative-gain curve: share of all High emails found when reading the top x% of the test set
order = np.argsort(-P[:, 0])
cum = np.cumsum(te.priority.values[order] == "High") / (te.priority == "High").sum()
pd.DataFrame({"frac_read": np.arange(1, len(cum) + 1) / len(cum), "frac_high_found": cum}).to_csv(
    f"{T}/cumulative_gain.csv", index=False)

# per-email test predictions (for the inbox demo)
out = te[["id", "utc", "sender", "subject", "priority", "resp_kind", "resp_hours"]].copy()
for j, c in enumerate(CLASSES):
    out[f"p_{c}"] = P[:, j]
out["pred"] = preds[best_key]
out["score"] = score
out.to_parquet(f"{OUT}/test_predictions.parquet")

# --- K-Means (unsupervised view of email "types")
Xw = df[NUM].astype(float).clip(df[NUM].quantile(0.01), df[NUM].quantile(0.99), axis=1)  # winsorize outliers
Xk = StandardScaler().fit_transform(Xw)
sil = []
for k in range(2, 11):
    km = KMeans(k, n_init=20, random_state=0).fit(Xk)
    sil.append(dict(k=k, silhouette=silhouette_score(Xk, km.labels_, sample_size=3000, random_state=0),
                    inertia=km.inertia_))
sil = pd.DataFrame(sil)
sil.to_csv(f"{T}/kmeans_selection.csv", index=False)
K = int(sil.loc[sil.k.between(3, 8)].sort_values("silhouette", ascending=False).iloc[0].k)
km = KMeans(K, n_init=50, random_state=0).fit(Xk)
df["cluster"] = km.labels_
pca = PCA(2, random_state=0).fit(Xk)
xy = pca.transform(Xk)
df["pc1"], df["pc2"] = xy[:, 0], xy[:, 1]
df[["id", "cluster", "pc1", "pc2", "priority"] + NUM].to_parquet(f"{OUT}/clusters.parquet")
json.dump({"k": K, "pca_var": pca.explained_variance_ratio_.tolist()}, open(f"{T}/kmeans_meta.json", "w"))
prof = df.groupby("cluster")[NUM].mean()
prof["size"] = df.cluster.value_counts().sort_index()
mix = pd.crosstab(df.cluster, df.priority, normalize="index")[CLASSES]
prof = prof.join(mix)
# top distinctive words per cluster
tf = TfidfVectorizer(min_df=5, max_df=0.4, stop_words="english", sublinear_tf=True)
Xt = tf.fit_transform(df.text)
voc = np.array(tf.get_feature_names_out())
glob = np.asarray(Xt.mean(0)).ravel()
prof["top_words"] = [", ".join(voc[np.argsort(-(np.asarray(Xt[df.cluster.values == c].mean(0)).ravel() - glob))[:8]])
                     for c in prof.index]
prof.to_csv(f"{T}/kmeans_profiles.csv")
print("K =", K); print(prof[["size"] + CLASSES + ["top_words"]].round(2).to_string())
