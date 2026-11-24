"""Stage 2 · Step 3 — Feature engineering (content, metadata, behavioural) + request-intent model.

Every behavioural feature is computed from information available *before* the email arrived
(no look-ahead), and folder membership is NOT a feature (folders are assigned after reading,
so they would leak the label).
"""
import os, re, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
import joblib

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
INTENT = os.path.join(HERE, "..", "data", "intent")
WINDOW_START, WINDOW_END, SPLIT = "1999-11-01", "2002-01-15", "2001-07-01"  # end: leave >=19 days to observe late replies

recv = pd.read_parquet(f"{OUT}/received_labeled.parquet")
sent = pd.read_parquet(f"{OUT}/sent.parquet")
OWNER = {"vince.kaminski@enron.com", "j.kaminski@enron.com", "vkaminski@aol.com", "kaminski@enron.com",
         "j..kaminski@enron.com", "vince.j.kaminski@enron.com", "vkamins@ect.enron.com", "vkaminski@palm.net"}

# ---------------------------------------------------------------- novel text (strip quoted history)
CUT = re.compile(r"(-{3,}\s*Original Message\s*-{3,}|-{5,}\s*Forwarded by|^\s*From:\s.*\n\s*(Sent|To):|"
                 r"^\s*>|^[^\n]{0,80}\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2} [AP]M\s*$)", re.M | re.I)


def novel(body):
    m = CUT.search(body)
    head = body[: m.start()] if m else body
    return head if len(head.strip()) >= 15 else body  # pure forwards: keep the forwarded content


recv["novel"] = recv.body.map(novel)
recv["text"] = (recv.subject + " . " + recv.novel).str.replace(r"\s+", " ", regex=True)

# ---------------------------------------------------------------- dataset 2: request-intent sentences
pos = [l.strip() for l in open(f"{INTENT}/intent_pos", encoding="utf-8", errors="ignore") if l.strip()]
neg = [l.strip() for l in open(f"{INTENT}/intent_neg", encoding="utf-8", errors="ignore") if l.strip()]
Xi, yi = pos + neg, [1] * len(pos) + [0] * len(neg)
intent = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                       LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
cv = cross_validate(intent, Xi, yi, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                    scoring=["accuracy", "f1", "roc_auc"])
intent_cv = {k: float(np.mean(v)) for k, v in cv.items() if k.startswith("test_")}
intent_cv.update(n_pos=len(pos), n_neg=len(neg))
print("intent model 5-fold CV:", intent_cv)
intent.fit(Xi, yi)
joblib.dump(intent, f"{OUT}/intent_model.joblib")
json.dump(intent_cv, open(f"{OUT}/tables/intent_cv.json", "w"), indent=2)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def intent_scores(txt):
    sents = [s.strip() for s in SENT_SPLIT.split(txt) if 12 <= len(s.strip()) <= 400][:40]
    if not sents:
        return 0.0, 0
    p = intent.predict_proba(sents)[:, 1]
    return float(p.max()), int((p > 0.5).sum())


sc = recv.novel.map(intent_scores)
recv["intent_max"] = sc.map(lambda t: t[0])
recv["intent_n"] = sc.map(lambda t: t[1])

# ---------------------------------------------------------------- content features
low = recv.text.str.lower()
recv["subj_len"] = recv.subject.str.len()
recv["body_words"] = recv.novel.str.split().str.len()
recv["log_body_words"] = np.log1p(recv.body_words)
recv["n_question"] = recv.novel.str.count(r"\?")
recv["n_exclaim"] = recv.novel.str.count("!")
recv["is_reply_subj"] = recv.subject.str.match(r"(?i)^\s*re\s*:").astype(int)
recv["is_fwd_subj"] = recv.subject.str.match(r"(?i)^\s*(fw|fwd)\s*:").astype(int)
recv["empty_subj"] = (recv.subject.str.len() == 0).astype(int)
recv["action_terms"] = low.str.count(r"\b(please|could you|can you|would you|let me know|need|send me|call me|"
                                     r"review|approve|confirm|respond|reply|schedule|meet)\b")
recv["deadline_terms"] = low.str.count(r"\b(asap|urgent|deadline|today|tomorrow|tonight|by (monday|tuesday|"
                                       r"wednesday|thursday|friday|eod|end of)|this week|next week|immediately)\b")
recv["date_refs"] = low.str.count(r"\b(\d{1,2}/\d{1,2}(/\d{2,4})?|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
                                  r"[a-z]*\.? \d{1,2}|\d{1,2}:\d{2}\s?(am|pm)?|monday|tuesday|wednesday|thursday|friday)\b")
recv["has_attachment"] = recv.body.str.contains(r"(?i)<<\s*File:|\.(doc|xls|ppt|pdf|zip)\b|\battach", regex=True).astype(int)
recv["has_url"] = recv.novel.str.contains(r"(?i)https?://|www\.", regex=True).astype(int)
recv["vince_named"] = recv.novel.str.contains(r"(?i)\bvince\b", regex=True).astype(int)
recv["unsubscribe"] = low.str.contains(r"unsubscribe|mailing list|newsletter|click here", regex=True).astype(int)

# ---------------------------------------------------------------- metadata features
recv["n_to"] = recv.to.map(len)
recv["n_cc"] = recv.cc.map(len)
recv["n_recip"] = recv.n_to + recv.n_cc
recv["owner_in_to"] = recv.to.map(lambda a: int(bool(OWNER & set(a))))
recv["owner_only_cc"] = ((recv.owner_in_to == 0) & recv.cc.map(lambda a: bool(OWNER & set(a)))).astype(int)
recv["sole_recipient"] = ((recv.n_recip == 1) & (recv.owner_in_to == 1)).astype(int)
loc = recv.utc.dt.tz_convert("US/Central")  # Enron HQ was in Houston
recv["hour"] = loc.dt.hour
recv["weekday"] = loc.dt.weekday
recv["weekend"] = (recv.weekday >= 5).astype(int)
recv["after_hours"] = ((recv.hour < 7) | (recv.hour >= 19)).astype(int)
recv["sender_internal"] = recv.sender.str.endswith("enron.com").astype(int)
recv["sender_edu"] = recv.sender.str.contains(r"\.edu$", regex=True).astype(int)
recv["sender_automated"] = recv.sender.str.contains(r"(?i)^(announce|enron\.announcements|perfmgmt|info|news|"
                                                    r"admin|mailer|listserv|noreply|no\.address)", regex=True).astype(int)
recv["bulk_flag"] = recv.bulk.astype(int)

# ---------------------------------------------------------------- behavioural features (past only)
recv = recv.sort_values("utc").reset_index(drop=True)
sent_events = [(t, a) for t, to, cc in zip(sent.utc, sent.to, sent.cc) for a in set(to) | set(cc)]
sent_events = pd.DataFrame(sent_events, columns=["utc", "addr"]).sort_values("utc")
sent_by_addr = {a: g.utc.values for a, g in sent_events.groupby("addr")}

prior_msgs, prior_reply_rate, prior_sent_to, days_since, thread_prior, owner_in_thread = [], [], [], [], [], []
hist = {}  # sender -> list of (arrival, response_time or None)
thread_hist = {}
sent_thread_times = {k: g.utc.values for k, g in sent.groupby("thread_id")}
for r in recv.itertuples():
    t = r.utc
    h = hist.get(r.sender, [])
    prior_msgs.append(len(h))
    # a past message counts as "answered" only if the answer had already happened by time t
    answered = sum(1 for (a, rt) in h if rt is not None and rt < t)
    prior_reply_rate.append((answered + 0.2) / (len(h) + 1))          # smoothed toward the global rate
    st = sent_by_addr.get(r.sender, np.array([], dtype="datetime64[ns]"))
    n_before = int(np.searchsorted(st, np.datetime64(t.tz_convert(None)), side="left")) if len(st) else 0
    prior_sent_to.append(n_before)
    last = [a for a, _ in h]
    last_sent = st[n_before - 1] if n_before else None
    cands = [x for x in [last[-1] if last else None] if x is not None]
    if last_sent is not None:
        cands.append(pd.Timestamp(last_sent).tz_localize("UTC"))
    days_since.append(min(365.0, (t - max(cands)).total_seconds() / 86400) if cands else 365.0)
    th = thread_hist.get(r.thread_id, 0)
    thread_prior.append(th)
    stt = sent_thread_times.get(r.thread_id, np.array([], dtype="datetime64[ns]"))
    owner_in_thread.append(int((stt < np.datetime64(t.tz_convert(None))).any()) if len(stt) else 0)
    resp = (t + pd.Timedelta(hours=r.resp_hours)) if r.resp_kind != "none" else None
    hist.setdefault(r.sender, []).append((t, resp))
    thread_hist[r.thread_id] = th + 1

recv["sender_prior_msgs"] = np.log1p(prior_msgs)
recv["sender_prior_reply_rate"] = prior_reply_rate
recv["owner_prior_sent_to_sender"] = np.log1p(prior_sent_to)
recv["days_since_contact"] = days_since
recv["thread_prior_msgs"] = thread_prior
recv["owner_in_thread_before"] = owner_in_thread

GROUPS = {
    "content": ["subj_len", "log_body_words", "n_question", "n_exclaim", "is_reply_subj", "is_fwd_subj", "empty_subj",
                "action_terms", "deadline_terms", "date_refs", "has_attachment", "has_url", "vince_named",
                "unsubscribe", "intent_max", "intent_n"],
    "metadata": ["n_to", "n_cc", "n_recip", "owner_in_to", "owner_only_cc", "sole_recipient", "hour", "weekday",
                 "weekend", "after_hours", "sender_internal", "sender_edu", "sender_automated", "bulk_flag"],
    "behavioral": ["sender_prior_msgs", "sender_prior_reply_rate", "owner_prior_sent_to_sender",
                   "days_since_contact", "thread_prior_msgs", "owner_in_thread_before"],
}
json.dump(GROUPS, open(f"{OUT}/feature_groups.json", "w"), indent=2)

recv["in_window"] = (recv.utc >= WINDOW_START) & (recv.utc < WINDOW_END)
recv["split"] = np.where(recv.utc < SPLIT, "train", "test")
keep = ["id", "utc", "sender", "subject", "text", "novel", "priority", "resp_kind", "resp_hours", "user_filed",
        "in_window", "split"] + sum(GROUPS.values(), [])
recv[keep].to_parquet(f"{OUT}/features.parquet")
# KNIME-ready export (numeric features + label, no raw text)
recv.loc[recv.in_window, ["id", "utc", "split", "priority"] + sum(GROUPS.values(), [])].to_csv(
    f"{OUT}/features_for_knime.csv", index=False)
w = recv[recv.in_window]
print(w.split.value_counts().to_dict())
print(pd.crosstab(w.split, w.priority, normalize="index").round(3))
print(w.groupby("priority")[["intent_max", "action_terms", "sender_prior_reply_rate", "owner_in_to", "n_recip"]].mean().round(3))
