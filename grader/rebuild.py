"""Rebuild the tracker from every check-in and submission issue.
Usage: python grader/rebuild.py issues.jsonl
Writes tracking/checkins.csv, tracking/submissions.csv and TRACKER.md.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from common import (clean_name, contest_title, ignored, is_kind, load_issues, local_time, login_of,
                    parse_form)
from config import CONTEST_ANSWERS, MIN_ENTRANTS_FOR_PRIZE, PROJECTS, WEEKS

issues = load_issues(sys.argv[1])

checkins, submissions = [], []
for issue in issues:
    login = login_of(issue)
    if not login or ignored(login):
        continue
    form = parse_form(issue.get("body"))
    project = form.get("Project", "")
    if project not in PROJECTS:
        continue
    row = {
        "issue": issue["number"],
        "login": login,
        "name": clean_name(form.get("Your name", ""), login),
        "project": project,
        "created_at": issue["created_at"],
    }
    if is_kind(issue, "check-in"):
        week, contest = form.get("Week", ""), form.get("Joining the model contest?", "")
        if week in WEEKS and contest in CONTEST_ANSWERS:
            checkins.append({**row, "week": week, "contest": contest})
    elif is_kind(issue, "submission"):
        labels = [l["name"] if isinstance(l, dict) else l for l in issue.get("labels") or []]
        submissions.append({**row, "valid": "valid" in labels})

ck = pd.DataFrame(checkins, columns=["issue", "login", "name", "project", "created_at", "week", "contest"])
sb = pd.DataFrame(submissions, columns=["issue", "login", "name", "project", "created_at", "valid"])
os.makedirs("tracking", exist_ok=True)
ck.sort_values("issue").to_csv("tracking/checkins.csv", index=False)
sb.sort_values("issue").to_csv("tracking/submissions.csv", index=False)

# TRACKER.md
now = local_time(datetime.now(timezone.utc)).strftime("%b %d, %Y at %I:%M %p")
lines = [
    "# Project Tracker",
    "",
    f"_Updated automatically on {now} (Detroit time) from check-in and submission issues._",
    "",
    "## Who checked in each week",
    "",
    "Number of different students per project and week.",
    "",
    "| Project | " + " | ".join(w.replace("Week ", "Wk ") for w in WEEKS) + " | Any week |",
    "|---|" + "---|" * (len(WEEKS) + 1),
]
total_week = {w: set() for w in WEEKS}
total_any = set()
for project in PROJECTS:
    p = ck[ck["project"] == project]
    cells = []
    for w in WEEKS:
        who = set(p.loc[p["week"] == w, "login"])
        total_week[w] |= {(project, x) for x in who}
        cells.append(str(len(who)))
    anyone = set(p["login"])
    total_any |= {(project, x) for x in anyone}
    lines.append(f"| {project} | " + " | ".join(cells) + f" | {len(anyone)} |")
lines.append("| **Total** | " + " | ".join(f"**{len(total_week[w])}**" for w in WEEKS)
             + f" | **{len(total_any)}** |")

lines += [
    "",
    f"## Model contest{contest_title()}",
    "",
    "Contest answer = each student's latest check-in. Valid files = students with at least one file that passed the format check.",
    "",
    f"| Project | Yes | Maybe | No | Valid files | Prize? (needs {MIN_ENTRANTS_FOR_PRIZE}+) |",
    "|---|---|---|---|---|---|",
]
latest = ck.sort_values("issue").groupby(["project", "login"]).tail(1)
for project in PROJECTS:
    answers = latest.loc[latest["project"] == project, "contest"].value_counts()
    yes, maybe, no = (int(answers.get(a, 0)) for a in CONTEST_ANSWERS)
    files = sb[(sb["project"] == project) & sb["valid"]]["login"].nunique()
    if files >= MIN_ENTRANTS_FOR_PRIZE:
        prize = "✅ Yes"
    elif files + yes >= MIN_ENTRANTS_FOR_PRIZE:
        prize = "Likely"
    elif files + yes + maybe >= MIN_ENTRANTS_FOR_PRIZE:
        prize = "Maybe"
    else:
        prize = "Not yet"
    lines.append(f"| {project} | {yes} | {maybe} | {no} | {files} | {prize} |")

lines += [
    "",
    "## Totals",
    "",
    f"- Check-ins recorded: {len(ck)}",
    f"- Contest files submitted: {len(sb)} ({int(sb['valid'].sum())} valid)",
    "",
]
with open("TRACKER.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"{len(ck)} check-ins, {len(sb)} submissions")
