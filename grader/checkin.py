"""Reply to one new check-in issue. Writes reply.md."""
from common import (clean_name, contest_day_text, ignored, load_event, login_of, parse_form,
                    write_output)
from config import CONTEST_ANSWERS, PROJECTS, WEEKS

issue = load_event()
login = login_of(issue)
form = parse_form(issue.get("body"))
project, week = form.get("Project", ""), form.get("Week", "")
contest = form.get("Joining the model contest?", "")
name = clean_name(form.get("Your name", ""), login)

problems = []
if project not in PROJECTS:
    problems.append("pick your **Project**")
if week not in WEEKS:
    problems.append("pick the **Week** you finished")
if contest not in CONTEST_ANSWERS:
    problems.append("answer **Joining the model contest?**")

if problems:
    reply = ("Could not record this check-in. Please open a new check-in with the form and "
             + ", ".join(problems) + ".")
    write_output("result", "invalid")
else:
    reply = f"Thanks, {name}! You're checked in for **{week}** of **{project}**."
    if contest == "Yes":
        reply += f" See you at the model evaluation day{contest_day_text()}!"
    elif contest == "Maybe":
        reply += f" We hope you can join the model evaluation{contest_day_text()}."
    if form.get("Anything blocking you? (optional)"):
        reply += "\n\nA project manager will read your note and get back to you."
    if ignored(login):
        reply += "\n\n_(Test account: this check-in won't be counted.)_"
    write_output("result", "valid")

with open("reply.md", "w", encoding="utf-8") as f:
    f.write(reply + "\n")
print(reply.splitlines()[0][:80])
