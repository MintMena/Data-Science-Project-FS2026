"""Shared helpers: read issue forms, download and check submission files, score them."""
import io
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import CONTEST_DAY, DEADLINE, IGNORE_LOGINS, PROJECTS, TIMEZONE

MAX_BYTES = 2_000_000
ATTACHMENT_URL = re.compile(
    r"https://github\.com/(?:user-attachments/files/\d+/[^\s)\]\"'<>]+"
    r"|[\w.-]+/[\w.-]+/files/\d+/[^\s)\]\"'<>]+)"
)


# issue forms
def parse_form(body):
    """Turn an issue-form body ('### Label' followed by the answer) into {label: answer}."""
    fields, label, lines = {}, None, []
    for line in (body or "").replace("\r\n", "\n").split("\n"):
        if line.startswith("### "):
            if label is not None:
                fields[label] = "\n".join(lines).strip()
            label, lines = line[4:].strip(), []
        elif label is not None:
            lines.append(line)
    if label is not None:
        fields[label] = "\n".join(lines).strip()
    return {k: ("" if v == "_No response_" else v) for k, v in fields.items()}


def clean_name(text, login):
    """Make a display name safe to show in Markdown tables."""
    text = re.sub(r"[`|@<>\[\]\\*_#]", "", text or "")
    text = " ".join(text.split())[:40]
    return text or login


def is_kind(issue, kind):
    labels = issue.get("labels") or []
    names = [l["name"] if isinstance(l, dict) else l for l in labels]
    prefix = {"check-in": "[Check-in]", "submission": "[Submission]"}[kind]
    return kind in names or (issue.get("title") or "").startswith(prefix)


def login_of(issue):
    user = issue.get("user")
    if isinstance(user, dict):
        return user.get("login", "")
    return issue.get("login", "") or ""


def ignored(login):
    return login.lower() in {x.lower() for x in IGNORE_LOGINS}


def load_event():
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as f:
        return json.load(f)["issue"]


def load_issues(path):
    """Read the JSON-lines file written by `gh api --paginate ... --jq '.[]'`."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def deadline():
    """The submission deadline, or None if it isn't set yet."""
    return datetime.fromisoformat(DEADLINE) if DEADLINE else None


def contest_day_text():
    """' on Dec 9' once the date is set, or ' (date coming soon)' while it's TBD."""
    day = (CONTEST_DAY or "").strip()
    return " (date coming soon)" if day.upper() in ("", "TBD") else f" on {day}"


def contest_title():
    """' (Dec 9)' once the date is set, or nothing while it's TBD."""
    day = (CONTEST_DAY or "").strip()
    return "" if day.upper() in ("", "TBD") else f" ({day})"


def local_time(dt):
    return dt.astimezone(ZoneInfo(TIMEZONE))


def write_output(name, value):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


# submission files
class SubmissionError(Exception):
    """A problem the student can fix. The message is shown to them."""


def download(url):
    if not ATTACHMENT_URL.fullmatch(url):
        raise SubmissionError("The file link doesn't look like a GitHub attachment.")
    headers = [{}]
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers.append({"Authorization": f"token {token}"})
    last = None
    for extra in headers:
        req = urllib.request.Request(url, headers={"User-Agent": "dsc-contest-bot", **extra})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise SubmissionError("The file is too big (over 2 MB). Did you upload the right file?")
            return data
        except urllib.error.HTTPError as e:
            last = e
    raise SubmissionError(f"I couldn't download your file (error {last.code}). Try attaching it again.")


def get_csv_bytes(file_field):
    """Find the attached CSV in the form answer, or accept CSV text pasted into the box."""
    urls = ATTACHMENT_URL.findall(file_field or "")
    csv_urls = [u for u in urls if u.lower().endswith(".csv")]
    if csv_urls:
        return download(csv_urls[-1])
    if urls:
        raise SubmissionError("The attached file isn't a `.csv` file.")
    text = (file_field or "").strip().strip("`").strip()
    if text.lower().startswith("prediction"):
        return text.encode("utf-8")
    raise SubmissionError("I couldn't find a CSV file. Drag `contest_submission.csv` into the box and wait for the link to appear before you click Create.")


def read_submission(data, project):
    """Check the file format. Returns a clean DataFrame or raises SubmissionError."""
    spec = PROJECTS[project]
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        raise SubmissionError("I couldn't read the file as a CSV.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.startswith("unnamed")]]
    missing = [c for c in spec["columns"] if c not in df.columns]
    if missing:
        found = ", ".join(f"`{c[:30]}`" for c in df.columns[:6]) or "none"
        need = " and ".join(f"`{c}`" for c in spec["columns"])
        raise SubmissionError(f"The file needs the column {need}. I found: {found}.")
    df = df[spec["columns"]]
    if len(df) != spec["rows"]:
        raise SubmissionError(f"The file has {len(df)} rows, but {project} needs {spec['rows']} (one per row of the hidden file, no extra rows).")
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise SubmissionError("Some values are empty or not numbers.")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise SubmissionError("Some values are infinite.")
    pred = df["prediction"]
    if spec["task"] == "multiclass":
        if not ((pred % 1 == 0) & pred.between(0, 9)).all():
            raise SubmissionError("`prediction` must be a whole number from 0 to 9 (the hand class).")
        df["prediction"] = pred.astype(int)
    if spec["task"] == "binary":
        if not pred.isin([0, 1]).all():
            raise SubmissionError("`prediction` must be 0 or 1.")
        if not df["probability"].between(0, 1).all():
            raise SubmissionError("`probability` must be between 0 and 1.")
        df["prediction"] = pred.astype(int)
    return df


# scoring 
def score(df, answers, project):
    """Return (main score, tie-break score) for one valid submission."""
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 mean_absolute_error, mean_squared_error, roc_auc_score)
    spec = PROJECTS[project]
    y = answers.to_numpy()
    pred = df["prediction"].to_numpy()
    if spec["task"] == "multiclass":
        return f1_score(y, pred, average="macro", zero_division=0), accuracy_score(y, pred)
    if spec["task"] == "binary":
        return roc_auc_score(y, df["probability"].to_numpy()), matthews_corrcoef(y, pred)
    return float(np.sqrt(mean_squared_error(y, pred))), mean_absolute_error(y, pred)
