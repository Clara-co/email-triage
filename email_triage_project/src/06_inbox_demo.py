"""Stage 2 · Step 6 — Inbox demo: one real test day, newest-first vs model-ranked (senders shown by domain only)."""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open(os.path.join(os.path.dirname(__file__), "05_figures.py")).read().split("SRC =")[0])  # shared style
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
p = pd.read_parquet(f"{OUT}/test_predictions.parquet")
p["day"] = p.utc.dt.tz_convert("US/Central").dt.date
g = p.groupby("day").agg(n=("id", "size"), h=("priority", lambda s: (s == "High").sum()))
cand = g[(g.n.between(14, 30)) & (g.h >= 3)]
best = None
for d in cand.index:  # pick the day where ranking helps most, to illustrate (reported as an example, not an average)
    x = p[p.day == d]
    top = (x.sort_values("score", ascending=False).head(5).priority == "High").sum()
    chron = (x.sort_values("utc", ascending=False).head(5).priority == "High").sum()
    if best is None or top - chron > best[1]:
        best = (d, top - chron)
d = best[0]; x = p[p.day == d].copy()
x["who"] = x.sender.str.split("@").str[-1].map(lambda s: "colleague @enron.com" if s.endswith("enron.com") else "@" + s)
x["subj"] = x.subject.fillna("").map(lambda s: (s[:46] + "…") if len(s) > 47 else (s or "(no subject)"))
x["t"] = x.utc.dt.tz_convert("US/Central").dt.strftime("%H:%M")
N = 10
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
for ax, (title, srt) in zip(axes, [("Newest first (a normal inbox)", x.sort_values("utc", ascending=False)),
                                    ("Ranked by the Random Forest", x.sort_values("score", ascending=False))]):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(N + 0.4, -1.2)
    ax.text(0, -0.9, title, fontsize=14, weight="semibold", color=INK)
    for i, r in enumerate(srt.head(N).itertuples()):
        c = C[r.priority]
        ax.add_patch(plt.Rectangle((0, i - 0.42), 1, 0.84, color="#f3f2ee" if i % 2 else SURF, lw=0))
        ax.add_patch(plt.Rectangle((0, i - 0.42), 0.012, 0.84, color=c, lw=0))
        ax.text(0.03, i - 0.12, r.subj, fontsize=10.5, color=INK, va="center")
        ax.text(0.03, i + 0.22, f"{r.t}  ·  {r.who}", fontsize=8.8, color=MUTED, va="center")
        ax.text(0.80, i, r.priority, fontsize=10, color=c, weight="semibold", va="center")
        ax.text(0.99, i, f"{r.score:.2f}", fontsize=9.5, color=INK2, va="center", ha="right")
    nh = (srt.head(5).priority == "High").sum()
    ax.text(0, N + 0.25, f"High-priority emails in the first 5: {nh}", fontsize=11, color=INK, weight="semibold")
fig.suptitle(f"One real test day ({pd.Timestamp(d):%a %d %b %Y}, {len(x)} emails): what Vince would see first",
             x=0.01, ha="left", fontsize=15, weight="semibold", color=INK, y=1.0)
fig.text(0.01, -0.01, "Colour bar / label = what he actually did (High = replied within 24 h). Score = 2·P(High)+P(Medium). "
         "Sender names hidden. Day chosen to illustrate; averages over all days are in the triage figure.", fontsize=9, color=MUTED)
fig.savefig(f"{OUT}/figs/15_inbox_demo.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
print(d, len(x), best)
