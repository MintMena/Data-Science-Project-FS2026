"""Score every student's latest valid submission against the hidden answers.

Usage: python grader/reveal.py issues.jsonl
Needs the HIDDEN_ANSWERS secret (made with pm_tools/make_answers_secret.py).
Writes leaderboard/results.csv and LEADERBOARD.md. Never prints the answers.
"""
import base64
import io
import os
import sys
import zipfile

import pandas as pd

from common import (SubmissionError, clean_name, contest_title, deadline, get_csv_bytes, ignored,
                    is_kind, load_issues, login_of, parse_form, parse_time, read_submission,
                    score)
from config import PROJECTS

secret = os.environ.get("HIDDEN_ANSWERS", "")
if not secret:
    sys.exit("The HIDDEN_ANSWERS secret is missing. See SETUP.md, step 6.")
answers = {}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(secret))) as z:
    for project, spec in PROJECTS.items():
        with z.open(f"{spec['key']}_hidden_answers.csv") as f:
            answers[project] = pd.read_csv(f)["answer"]
        if len(answers[project]) != spec["rows"]:
            sys.exit(f"The answer file for {project} has the wrong number of rows.")

# latest valid submission per student and project, made before the deadline
latest = {}
for issue in load_issues(sys.argv[1]):
    login = login_of(issue)
    labels = [l["name"] if isinstance(l, dict) else l for l in issue.get("labels") or []]
    if not is_kind(issue, "submission") or "valid" not in labels or not login or ignored(login):
        continue
    if deadline() and parse_time(issue["created_at"]) > deadline():
        continue
    form = parse_form(issue.get("body"))
    project = form.get("Project", "")
    if project not in PROJECTS:
        continue
    key = (project, login)
    if key not in latest or issue["number"] > latest[key][0]["number"]:
        latest[key] = (issue, form)

rows = []
for (project, login), (issue, form) in latest.items():
    try:
        df = read_submission(get_csv_bytes(form.get("Your file", "")), project)
        main, tie = score(df, answers[project], project)
    except SubmissionError as e:
        print(f"Skipped issue #{issue['number']}: {e}")
        continue
    rows.append({
        "project": project,
        "name": clean_name(form.get("Your name", ""), login),
        "login": login,
        "score": main,
        "tiebreak": tie,
        "issue": issue["number"],
        "submitted_at": issue["created_at"],
    })

res = pd.DataFrame(rows, columns=["project", "name", "login", "score", "tiebreak", "issue", "submitted_at"])
ranked = []
for project, spec in PROJECTS.items():
    p = res[res["project"] == project].copy()
    up = spec["higher_is_better"]
    p = p.sort_values(["score", "tiebreak", "issue"], ascending=[not up, not up, True])
    p.insert(1, "rank", range(1, len(p) + 1))
    p["metric"], p["tiebreak_metric"] = spec["metric"], spec["tiebreak"]
    ranked.append(p)
res = pd.concat(ranked) if ranked else res
os.makedirs("leaderboard", exist_ok=True)
res.to_csv("leaderboard/results.csv", index=False)

lines = [f"# Model Contest Leaderboard{contest_title()}", ""]
for project, spec in PROJECTS.items():
    p = res[res["project"] == project]
    arrow = "higher is better" if spec["higher_is_better"] else "lower is better"
    lines += [f"## {project}", "", f"Main score: **{spec['metric']}** ({arrow}). Ties are broken by {spec['tiebreak']}.", ""]
    if p.empty:
        lines += ["_No valid submissions._", ""]
        continue
    lines += [f"| Rank | Name | {spec['metric']} | {spec['tiebreak']} |", "|---|---|---|---|"]
    for _, r in p.iterrows():
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r["rank"], "")
        lines.append(f"| {medal} {r['rank']} | {r['name']} | {r['score']:.4f} | {r['tiebreak']:.4f} |")
    lines.append("")
with open("LEADERBOARD.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Scored {len(res)} submissions.")
