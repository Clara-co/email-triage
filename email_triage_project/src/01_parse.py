"""Stage 2 · Step 1 — Parse the kaminski-v mailbox into a tidy table.

Input : enron-memory-replay (Akadal, 2026) manifest + raw message text, which is a
        de-duplicated copy of the kaminski-v mailbox from the CMU Enron release (2015-05-07).
Output: out/emails.parquet  (one row per unique message, sent and received)
"""
import glob, os, re, sys
from email import message_from_string
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "emr")
OUT = os.path.join(os.path.dirname(__file__), "..", "out")

man = pd.read_csv(f"{ROOT}/dataset/manifest.csv").rename(columns={"class": "cls"})
files = {os.path.basename(p)[:-3]: p for p in glob.glob(f"{ROOT}/memory/archive/mail/*/*.md")}

FENCE = re.compile(r"```text\n(.*?)\n```", re.S)
ADDR = re.compile(r"[\w.\-+'&]+@[\w.\-]+")


def addrs(s):
    return sorted(set(a.lower().strip(".") for a in ADDR.findall(s or "")))


rows = []
for r in man.itertuples():
    txt = open(files[r.id], encoding="utf-8").read()
    raw = FENCE.search(txt).group(1)
    msg = message_from_string(raw)
    body = msg.get_payload()
    if not isinstance(body, str):
        body = str(body)
    folders = {p.split("/")[2] for p in [r.path] + (r.alias_paths.split() if isinstance(r.alias_paths, str) else [])}
    rows.append(dict(
        id=r.id, utc=pd.Timestamp(r.utc), direction=r.direction, bulk=r.cls == "bulk",
        thread_id=r.thread_id, quoted_from=r.quoted_from if isinstance(r.quoted_from, str) else "",
        sender=(addrs(msg.get("From")) or [""])[0], to=addrs(msg.get("To")), cc=addrs(msg.get("Cc")),
        x_to=msg.get("X-To", ""), x_from=msg.get("X-From", ""),
        subject=re.sub(r"\s+", " ", msg.get("Subject", "") or "").strip(),
        body=body, folders=sorted(folders), novel_chars=r.novel_chars, chars=r.chars,
    ))

df = pd.DataFrame(rows).sort_values("utc").reset_index(drop=True)
df.to_parquet(f"{OUT}/emails.parquet")
print(df.shape, df.direction.value_counts().to_dict())
print(df[["id", "utc", "sender", "subject"]].head())
