"""Stage 2 · Step 2 — Derive attention-priority labels from the owner's own behaviour.

The Enron corpus has no "importance" field, so we use *implicit feedback*: what Vince Kaminski
actually did with each message he received (cf. Dabbish et al., 2005; Aberdeen et al., 2010).

  HIGH    he replied to or forwarded the message within 24 hours   -> needs attention AND time-sensitive
  MEDIUM  he replied/forwarded, but only after 24 hours            -> needs attention, not urgent
  LOW     he never replied to or forwarded it                      -> did not need his attention

A folder-filing variant (filed into one of his own folders -> Medium) was tried and rejected: ~75% of
mass mailings counted as "filed", and folder data stops at the June-2001 export (see report §3).

A "response" is a message he sent later that (a) quotes the received message (quote links in the
manifest) or (b) is in the same thread, is addressed to the original sender and starts with RE/FW,
within 14 days.
"""
import os
import re
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
df = pd.read_parquet(f"{OUT}/emails.parquet")

OWNER = {"vince.kaminski@enron.com", "j.kaminski@enron.com", "vkaminski@aol.com", "kaminski@enron.com",
         "j..kaminski@enron.com", "vince.j.kaminski@enron.com", "vkamins@ect.enron.com", "vkaminski@palm.net"}
SYSTEM_FOLDERS = {"all_documents", "discussion_threads", "deleted_items", "inbox", "notes_inbox",
                  "_sent_mail", "sent", "sent_items", "calendar"}

RESP_RE = re.compile(r"(?i)^\s*(re|fw|fwd)\s*:")
FWD_RE = re.compile(r"(?i)^\s*(fw|fwd)\s*:")

sent = df[df.direction == "sent"].copy()
recv = df[(df.direction == "received") & (~df.sender.isin(OWNER))].copy()

# index the owner's sent mail by the messages it quotes
quoted_by = {}
for s in sent.itertuples():
    for q in s.quoted_from.split():
        quoted_by.setdefault(q, []).append((s.utc, s.subject))
sent_by_thread = {k: g for k, g in sent.groupby("thread_id")}

resp_time, resp_kind = [], []
for r in recv.itertuples():
    cands = [(t, subj) for t, subj in quoted_by.get(r.id, []) if t > r.utc]
    if not cands and r.thread_id in sent_by_thread:  # fallback: same thread, addressed back to sender
        g = sent_by_thread[r.thread_id]
        cands = [(x.utc, x.subject) for x in g.itertuples()
                 if r.utc < x.utc <= r.utc + pd.Timedelta(days=14)
                 and RESP_RE.match(x.subject) and r.sender in (list(x.to) + list(x.cc))]
    if cands:
        t, subj = min(cands)
        resp_time.append((t - r.utc).total_seconds() / 3600)
        resp_kind.append("forward" if FWD_RE.match(subj) else "reply")
    else:
        resp_time.append(np.nan); resp_kind.append("none")

recv["resp_hours"] = resp_time
recv["resp_kind"] = resp_kind
recv["user_filed"] = recv.folders.apply(lambda fs: any(f not in SYSTEM_FOLDERS for f in fs))
recv["deleted"] = recv.folders.apply(lambda fs: "deleted_items" in fs)

def label(r):
    if r.resp_kind == "none":
        return "Low"
    return "High" if r.resp_hours <= 24 else "Medium"

recv["priority"] = recv.apply(label, axis=1)
recv.to_parquet(f"{OUT}/received_labeled.parquet")
sent.to_parquet(f"{OUT}/sent.parquet")

print("received (non-self):", len(recv))
print(recv.priority.value_counts())
print(recv.resp_kind.value_counts())
print("median reply latency (h):", recv.resp_hours.median())
print(pd.crosstab(recv.priority, recv.user_filed))
