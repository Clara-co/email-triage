"""Stage 2 · Step 5 — Figures (static PNGs for the report and video)."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.tree import plot_tree, DecisionTreeClassifier

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
T, F = f"{OUT}/tables", f"{OUT}/figs"
for f in ["/usr/share/fonts/opentype/inter/Inter-Regular.otf", "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
          "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", "/usr/share/fonts/opentype/inter/Inter-Bold.otf"]:
    if os.path.exists(f):
        fm.fontManager.addfont(f)

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8d85", "#e6e5e0"
C = {"High": "#eb6834", "Medium": "#2a78d6", "Low": "#1baf7a"}
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CLASSES = ["High", "Medium", "Low"]

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "semibold",
    "axes.titlesize": 14, "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 14,
    "legend.frameon": False, "xtick.major.size": 0, "ytick.major.size": 0, "figure.dpi": 100,
})


def save(fig, name, note=None):
    if note:
        fig.text(0.01, -0.02, note, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.savefig(f"{F}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def subtitle(ax, s):
    ax.text(0, 1.02, s, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")


SRC = "Data: Enron Email Dataset (CMU, 2015), kaminski-v mailbox via Akadal (2026)."
recv = pd.read_parquet(f"{OUT}/received_labeled.parquet")
feat = pd.read_parquet(f"{OUT}/features.parquet")
win = feat[feat.in_window]
res = pd.read_csv(f"{T}/model_comparison.csv")
G = json.load(open(f"{OUT}/feature_groups.json"))

# ------------------------------------------------------------------ 1 pipeline overview
fig, ax = plt.subplots(figsize=(13, 3.4)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 3.4)
steps = [("1  Data", "Enron kaminski-v\n11,523 unique msgs\n+ Parakweet intent\n3,655 sentences"),
         ("2  Labels", "Implicit feedback:\nreply ≤ 24 h → High\nlater reply → Medium\nno response → Low"),
         ("3  Features", "36 engineered\n(content, metadata,\nbehavioural) +\nTF-IDF text"),
         ("4  Models", "Logistic Regression\nDecision Tree\nRandom Forest\nNaive Bayes · K-Means"),
         ("5  Evaluate", "Chronological test\n(Jul 2001–Jan 2002)\nF1 · confusion ·\ninbox ranking NDCG")]
for i, (h, b) in enumerate(steps):
    x = 0.15 + i * 2.6
    ax.add_patch(FancyBboxPatch((x, 0.25), 2.25, 2.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#eef4fc" if i != 3 else "#fdeee7", ec=BLUE if i != 3 else ORANGE, lw=1.4))
    ax.text(x + 0.15, 2.72, h, fontsize=13, weight="semibold", color=INK, va="center")
    ax.text(x + 0.15, 1.45, b, fontsize=10.5, color=INK2, va="center", linespacing=1.5)
    if i < 4:
        ax.annotate("", xy=(x + 2.55, 1.65), xytext=(x + 2.28, 1.65),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
save(fig, "01_pipeline")

# ------------------------------------------------------------------ 2 monthly volume by label, with window
m = recv.set_index("utc").groupby([pd.Grouper(freq="ME"), "priority"]).size().unstack().fillna(0)[CLASSES]
m.index = m.index.tz_convert(None) - pd.Timedelta(days=15)  # centre bars on their month
fig, ax = plt.subplots(figsize=(12, 4.8))
bottom = np.zeros(len(m)); w = 24
for c in CLASSES:
    ax.bar(m.index, m[c], width=w, bottom=bottom, color=C[c], label=c, edgecolor=SURF, linewidth=1.2)
    bottom += m[c].values
ntr, nte = (win.split == "train").sum(), (win.split == "test").sum()
ax.axvspan(pd.Timestamp("1999-11-20"), pd.Timestamp("2001-07-01"), color=BLUE, alpha=0.06, lw=0)
ax.axvspan(pd.Timestamp("2001-07-01"), pd.Timestamp("2002-01-15"), color=ORANGE, alpha=0.08, lw=0)
ymax = bottom.max()
ax.text(pd.Timestamp("2000-09-01"), ymax * 0.84, f"TRAIN  ({ntr:,} emails)", ha="center", fontsize=10, color=BLUE, weight="semibold")
ax.text(pd.Timestamp("2001-10-08"), ymax * 1.07, f"TEST  ({nte:,})", ha="center", fontsize=10, color=ORANGE, weight="semibold")
ax.set_ylim(0, ymax * 1.15)
m_resp = 1 - recv.priority.eq("Low").mean(); ax.set_ylabel("Received emails per month"); ax.grid(axis="x", visible=False)
ax.set_title("Received mail per month, by derived priority label", pad=28)
subtitle(ax, "Chronological split: learn from the past, predict the future. Last 3 weeks dropped so late replies can be observed.")
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
save(fig, "02_volume_labels", SRC)

# ------------------------------------------------------------------ 3 reply latency
lat = recv.loc[recv.resp_kind != "none", "resp_hours"].clip(lower=0.05)
fig, ax = plt.subplots(figsize=(10, 4.4))
bins = np.logspace(np.log10(0.05), np.log10(lat.max()), 45)
n, b, patches = ax.hist(lat, bins=bins, color=BLUE, edgecolor=SURF, linewidth=1)
for p, left in zip(patches, b[:-1]):
    if left >= 24:
        p.set_facecolor(SEQ[1])
ax.set_xscale("log"); ax.axvline(24, color=INK, lw=1.2, ls=(0, (4, 3)))
share = (lat <= 24).mean()
ax.text(26, n.max() * 0.92, f"24-hour threshold\n{share:.0%} of responses are faster → 'High'", fontsize=10, color=INK)
ax.set_xticks([0.1, 1, 6, 24, 24 * 7, 24 * 30, 24 * 180]); ax.set_xticklabels(["6 min", "1 h", "6 h", "1 day", "1 wk", "1 mo", "6 mo"])
ax.set_ylabel("Responses"); ax.grid(axis="x", visible=False)
ax.set_title("How quickly did Vince Kaminski respond?", pad=28)
subtitle(ax, f"Time from arrival to his reply/forward, {len(lat):,} responded-to emails (log scale). Median {lat.median():.1f} h.")
save(fig, "03_reply_latency", SRC)

# ------------------------------------------------------------------ 4 feature signals by class (small multiples)
show = [("intent_max", "Request-intent score (max sentence)"), ("sole_recipient", "Vince is the only recipient"),
        ("owner_in_to", "Vince in To: (not just Cc:)"), ("is_reply_subj", "Subject starts with 'Re:'"),
        ("sender_prior_reply_rate", "Sender's past reply rate"), ("owner_in_thread_before", "Vince already in thread"),
        ("bulk_flag", "Tagged as bulk mail"), ("sender_internal", "Sender inside Enron")]
fig, axes = plt.subplots(2, 4, figsize=(14, 5.6))
for ax, (col, lab) in zip(axes.ravel(), show):
    vals = win.groupby("priority")[col].median() if col == "n_recip" else win.groupby("priority")[col].mean()
    vals = vals[CLASSES]
    ax.barh(range(3), vals.values, color=[C[c] for c in CLASSES], height=0.55)
    for i, v in enumerate(vals.values):
        ax.text(v, i, f"  {v:.2f}" if v < 10 else f"  {v:.0f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES); ax.invert_yaxis()
    ax.set_title(lab, fontsize=10.5, pad=6); ax.set_xlim(0, vals.max() * 1.35); ax.grid(axis="y", visible=False)
    ax.set_xticks([])
fig.suptitle(f"What distinguishes High-priority mail?  Feature averages by class (n = {len(win):,})",
             x=0.01, ha="left", fontsize=14, weight="semibold", color=INK, y=1.02)
fig.tight_layout()
save(fig, "04_feature_signals", SRC)

# ------------------------------------------------------------------ 5 model comparison (dot + 95% CI)
r = res.sort_values("macro_f1")
fig, ax = plt.subplots(figsize=(10, 4.6))
y = np.arange(len(r))
best = r.macro_f1.max()
for i, row in enumerate(r.itertuples()):
    is_base = row.model.startswith("Baseline")
    col = MUTED if is_base else (ORANGE if row.macro_f1 == best else BLUE)
    ax.plot([row.macro_f1_lo, row.macro_f1_hi], [i, i], color=col, lw=2, solid_capstyle="round")
    ax.scatter(row.macro_f1, i, s=90, color=col, zorder=3, edgecolor=SURF, linewidth=2)
    ax.text(row.macro_f1_hi + 0.01, i, f"{row.macro_f1:.3f}", va="center", fontsize=10.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(r.model); ax.set_xlim(0.2, 0.6)
ax.set_xlabel("Macro-averaged F1 on the 2001 test set (bar = bootstrap 95% CI)"); ax.grid(axis="y", visible=False)
ax.set_title("Random Forest is the strongest of the four classifiers", pad=28)
subtitle(ax, "Gray = baselines. Only the Random Forest clearly beats a hand-written rule set.")
save(fig, "05_model_comparison", SRC)

# ------------------------------------------------------------------ 6 confusion matrices
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + SEQ)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, (k, nm) in zip(axes, [("lr", "Logistic Regression"), ("dt", "Decision Tree"), ("rf", "Random Forest"), ("nb", "Naive Bayes")]):
    cm = pd.read_csv(f"{T}/cm_{k}.csv", index_col=0).values
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=10,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.grid(False); ax.set_title(nm, fontsize=12.5)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("Actual")
fig.suptitle("Confusion matrices (row-normalised: share of each true class)", x=0.01, ha="left", fontsize=14,
             weight="semibold", color=INK, y=1.04)
fig.tight_layout()
save(fig, "06_confusion_matrices", SRC)

# ------------------------------------------------------------------ 7 per-class F1 heatmap-table
models = [res.model.iloc[0], "Baseline: hand-written rules", "Logistic Regression", "Decision Tree",
          "Random Forest", "Naive Bayes (Complement)"]
cols = ["High_precision", "High_recall", "High_f1", "Medium_f1", "Low_f1", "accuracy", "macro_f1", "kappa"]
lab = ["High\nprecision", "High\nrecall", "High\nF1", "Medium\nF1", "Low\nF1", "Accuracy", "Macro\nF1", "Cohen's\nκ"]
M = res.set_index("model").loc[models, cols]
fig, ax = plt.subplots(figsize=(12, 4.3))
ax.imshow(M.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        colmax = np.nanmax(M.values[2:, j])
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                weight="bold" if (i >= 2 and v == colmax) else "normal", color="white" if v > 0.6 else INK)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=10); ax.xaxis.tick_top()
ax.set_yticks(range(len(models))); ax.set_yticklabels(models); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Test-set metrics by model (bold = best of the four classifiers)", pad=46)
save(fig, "07_metrics_table", SRC)

# ------------------------------------------------------------------ 8 ablation
ab = pd.read_csv(f"{T}/ablation.csv")
order = ["Behavioral only", "Text only (TF-IDF)", "Content (text + content features)", "Metadata only",
         "Metadata + Behavioral", "All features"]
fig, ax = plt.subplots(figsize=(10, 4.4))
for k, (mdl, col, off) in enumerate([("Logistic Regression", BLUE, -0.15), ("Random Forest", ORANGE, 0.15)]):
    s = ab[ab.model == mdl].set_index("features").loc[order]
    ax.barh(np.arange(6) + off, s.macro_f1, height=0.28, color=col, label=mdl)
    for i, v in enumerate(s.macro_f1):
        ax.text(v + 0.005, i + off, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(range(6)); ax.set_yticklabels(order); ax.set_xlim(0, 0.6); ax.grid(axis="y", visible=False)
ax.axvline(res.loc[res.model == "Baseline: hand-written rules", "macro_f1"].iloc[0], color=MUTED, ls=(0, (4, 3)), lw=1)
ax.text(res.loc[res.model == "Baseline: hand-written rules", "macro_f1"].iloc[0], -0.75, " rules baseline", color=MUTED, fontsize=9)
ax.set_xlabel("Macro-F1 on test set"); ax.legend(loc="lower right")
ax.set_title("The words carry the most signal, but combining every group works best", pad=28)
subtitle(ax, "Feature-group ablation: each model retrained on a subset of features with its tuned hyper-parameters.")
save(fig, "08_ablation", SRC)

# ------------------------------------------------------------------ 9 permutation importance
pi = pd.read_csv(f"{T}/perm_importance_rf.csv").head(15).iloc[::-1]
gcol = {"metadata": BLUE, "behavioral": ORANGE, "content": AQUA, "text (TF-IDF)": VIOLET}
pretty = {"n_recip": "Number of recipients", "n_to": "Number of To: recipients", "n_cc": "Number of Cc: recipients",
          "sender_prior_reply_rate": "Sender's past reply rate", "owner_in_to": "Vince in To:",
          "sole_recipient": "Vince sole recipient", "sender_prior_msgs": "Messages from sender so far",
          "owner_prior_sent_to_sender": "Vince's past mails to sender", "days_since_contact": "Days since last contact",
          "thread_prior_msgs": "Earlier messages in thread", "owner_in_thread_before": "Vince already in thread",
          "intent_max": "Request-intent score", "intent_n": "Request sentences", "log_body_words": "Body length",
          "subj_len": "Subject length", "hour": "Hour sent", "weekday": "Day of week", "text": "Email text (TF-IDF)",
          "sender_internal": "Sender inside Enron", "is_reply_subj": "Subject starts 'Re:'", "is_fwd_subj": "Subject 'Fw:'",
          "action_terms": "Action terms", "date_refs": "Date/time mentions", "has_attachment": "Attachment",
          "n_question": "Question marks", "bulk_flag": "Bulk-mail flag", "sender_edu": "Sender .edu",
          "has_url": "Contains URL", "vince_named": "Greets 'Vince'", "deadline_terms": "Deadline terms",
          "owner_only_cc": "Vince only in Cc:", "after_hours": "After hours", "unsubscribe": "List/unsubscribe terms",
          "sender_automated": "Automated sender", "n_exclaim": "Exclamation marks", "weekend": "Weekend", "empty_subj": "Empty subject"}
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.barh(range(len(pi)), pi.importance, xerr=pi.sd, color=[gcol[g] for g in pi.group], height=0.62,
        error_kw=dict(ecolor=MUTED, lw=1, capsize=0))
ax.set_yticks(range(len(pi))); ax.set_yticklabels([pretty.get(f, f) for f in pi.feature]); ax.grid(axis="y", visible=False)
ax.set_xlabel("Drop in test macro-F1 when the feature is shuffled")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v, label=k) for k, v in gcol.items()], loc="lower right", title="Feature group",
          title_fontsize=9.5)
ax.set_title("Random Forest: the 15 most important features", pad=28)
subtitle(ax, "Permutation importance on the 2001 test set (5 repeats; whiskers = ±1 SD).")
save(fig, "09_permutation_importance", SRC)

# ------------------------------------------------------------------ 10 LR words for High
co = pd.read_csv(f"{T}/lr_coefficients.csv", index_col=0)
wd = co[co.kind == "word"]["High"].sort_values()
top = pd.concat([wd.head(12), wd.tail(12)])
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.barh(range(len(top)), top.values, color=[BLUE if v > 0 else MUTED for v in top.values], height=0.65)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=10); ax.axvline(0, color=INK2, lw=0.8)
ax.grid(axis="y", visible=False); ax.set_xlabel("Logistic-regression coefficient for the High class")
ax.set_title("Words that push an email toward / away from 'High'", pad=28)
subtitle(ax, "Blue: raises the odds of High. Gray: lowers them. (Names of people are left in; the corpus is public.)")
save(fig, "10_lr_words", SRC)

# ------------------------------------------------------------------ 11 shallow decision tree (for interpretation)
NUM = sum(G.values(), [])
tr = win[win.split == "train"]
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, class_weight="balanced", random_state=0).fit(
    tr[NUM], tr.priority.astype(object))
fig, ax = plt.subplots(figsize=(17, 7))
art = plot_tree(dt, feature_names=[pretty.get(f, f) for f in NUM], class_names=list(dt.classes_), filled=False,
                impurity=False, proportion=True, rounded=True, fontsize=10, ax=ax, precision=2)
for a in art:
    txt = a.get_text()
    cls = txt.split("class = ")[-1].strip() if "class = " in txt else None
    if a.get_bbox_patch() is not None and cls in C:
        a.get_bbox_patch().set_facecolor(C[cls] + "33"); a.get_bbox_patch().set_edgecolor(C[cls])
ax.set_title("A 3-level decision tree, readable as triage rules (numeric features only)", pad=10)
save(fig, "11_decision_tree", SRC)

# ------------------------------------------------------------------ 12 triage evaluation
tri = pd.read_csv(f"{T}/triage_days.csv")
cg = pd.read_csv(f"{T}/cumulative_gain.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
data = [tri.ndcg_chrono, tri.ndcg_random, tri.ndcg_model]
names = ["Newest first\n(normal inbox)", "Random\norder", "Model-ranked\n(Random Forest)"]
cols = [MUTED, MUTED, ORANGE]
rng = np.random.default_rng(1)
for i, (d, c) in enumerate(zip(data, cols)):
    a1.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=14, color=c, alpha=0.55, lw=0)
    a1.plot([i - 0.28, i + 0.28], [d.mean()] * 2, color=INK, lw=2.2)
    a1.text(i + 0.32, d.mean(), f"{d.mean():.2f}", va="center", fontsize=11, weight="semibold", color=INK)
a1.set_xticks(range(3)); a1.set_xticklabels(names); a1.set_ylim(0, 1.04); a1.grid(axis="x", visible=False)
a1.set_ylabel("NDCG@10 per day"); a1.set_title(f"Ranking each day's inbox ({len(tri)} test days with 5+ emails)", fontsize=12.5)
a2.plot(cg.frac_read * 100, cg.frac_high_found * 100, color=ORANGE, lw=2, label="Read in model order")
a2.plot([0, 100], [0, 100], color=MUTED, lw=1.5, ls=(0, (4, 3)), label="Read in random order")
for x in [20, 40]:
    yv = np.interp(x / 100, cg.frac_read, cg.frac_high_found) * 100
    a2.scatter(x, yv, color=ORANGE, s=50, zorder=3, edgecolor=SURF, linewidth=2)
    a2.text(x + 2, yv - 6, f"top {x}% → {yv:.0f}% of High mail", fontsize=10, color=INK)
a2.set_xlabel("% of test emails read (sorted by P(High))"); a2.set_ylabel("% of all High emails found")
a2.set_title("Cumulative gain for High-priority mail", fontsize=12.5); a2.legend(loc="lower right"); a2.set_xlim(0, 100); a2.set_ylim(0, 102)
fig.suptitle("Does model ranking help a reader with an unread backlog?", x=0.01, ha="left", fontsize=14,
             weight="semibold", color=INK, y=1.03)
fig.tight_layout()
save(fig, "12_triage", SRC)

# ------------------------------------------------------------------ 13 PR curve High vs rest
P = np.load(f"{OUT}/test_probas.npy", allow_pickle=True).item()
te = win[win.split == "test"]
yh = (te.priority == "High").values
fig, ax = plt.subplots(figsize=(8, 5))
for k, nm, col in [("rf", "Random Forest", ORANGE), ("lr", "Logistic Regression", BLUE), ("dt", "Decision Tree", AQUA), ("nb", "Naive Bayes", VIOLET)]:
    p, rc, _ = precision_recall_curve(yh, P[k][:, 0])
    ap = res.loc[res.model.str.startswith(nm.split()[0]), "High_pr_auc"].iloc[0]
    ax.plot(rc, p, color=col, lw=2, label=f"{nm}  (AP {ap:.2f})")
ax.axhline(yh.mean(), color=MUTED, ls=(0, (4, 3)), lw=1.2); ax.text(0.62, yh.mean() + 0.015, f"prevalence {yh.mean():.2f}", color=MUTED, fontsize=9.5)
ax.set_xlabel("Recall (High)"); ax.set_ylabel("Precision (High)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper right"); ax.set_title("High vs. rest: precision–recall trade-off", pad=28)
subtitle(ax, "Every model ranks High mail well above chance (dashed line); none separates it cleanly.")
save(fig, "13_pr_high", SRC)

# ------------------------------------------------------------------ 14 K-Means profiles
prof = pd.read_csv(f"{T}/kmeans_profiles.csv", index_col=0)
sel = pd.read_csv(f"{T}/kmeans_selection.csv")
CL_NAMES = json.load(open(f"{T}/kmeans_names.json")) if os.path.exists(f"{T}/kmeans_names.json") else {}
prof = prof.sort_values("High")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[0.7, 2]))
a1.plot(sel.k, sel.silhouette, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=2)
kk = json.load(open(f"{T}/kmeans_meta.json"))["k"]
a1.scatter([kk], sel.set_index("k").loc[kk, "silhouette"], s=140, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=2)
a1.set_xlabel("k (number of clusters)"); a1.set_ylabel("Silhouette score"); a1.set_title(f"Choosing k (chosen k = {kk})", fontsize=12.5)
left = np.zeros(len(prof))
for c in CLASSES:
    a2.barh(range(len(prof)), prof[c] * 100, left=left, color=C[c], label=c, edgecolor=SURF, linewidth=1.5, height=0.62)
    for i, (l, v) in enumerate(zip(left, prof[c] * 100)):
        if v >= 7:
            a2.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color="white")
    left += prof[c].values * 100
a2.set_yticks(range(len(prof)))
a2.set_yticklabels([f"{CL_NAMES.get(str(i), 'Cluster ' + str(i))}  (n={int(n)})" for i, n in zip(prof.index, prof["size"])])
a2.set_xlim(0, 100); a2.grid(axis="y", visible=False); a2.set_xlabel("Share of cluster (%)")
a2.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0)); a2.set_title("Priority mix inside each K-Means cluster", fontsize=12.5)
fig.suptitle("Unsupervised view: K-Means finds kinds of email with very different priority profiles", x=0.01,
             ha="left", fontsize=14, weight="semibold", color=INK, y=1.04)
fig.tight_layout()
save(fig, "14_kmeans", SRC)
print("figures:", sorted(os.listdir(F)))
