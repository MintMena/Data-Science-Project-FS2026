"""Check the format of one contest submission. Writes reply.md and sets the result output."""
from common import (SubmissionError, clean_name, contest_day_text, deadline, get_csv_bytes,
                    ignored, load_event, local_time, login_of, parse_form, parse_time,
                    read_submission, write_output)
from config import PROJECTS

issue = load_event()
login = login_of(issue)
form = parse_form(issue.get("body"))
project = form.get("Project", "")
name = clean_name(form.get("Your name", ""), login)
created = parse_time(issue["created_at"])

try:
    if project not in PROJECTS:
        raise SubmissionError("Pick your **Project** in the form.")
    if deadline() and created > deadline():
        when = local_time(deadline()).strftime("%b %d at %I:%M %p")
        raise SubmissionError(f"Submissions closed on {when}.")
    df = read_submission(get_csv_bytes(form.get("Your file", "")), project)
    reply = (f"✅ Got it, {name}! Your **{project}** file has the correct format "
             f"({len(df)} rows, columns: {', '.join(f'`{c}`' for c in df.columns)}).\n\n"
             f"Scores are revealed at the model evaluation meeting{contest_day_text()}. If you submit again, "
             "your latest valid file counts.")
    if ignored(login):
        reply += "\n\n_(Test account: this submission won't be counted.)_"
    result = "valid"
except SubmissionError as e:
    reply = (f"❌ Your file wasn't accepted: {e}\n\n"
             "Please fix it and submit again with a new form.")
    result = "invalid"

write_output("result", result)
with open("reply.md", "w", encoding="utf-8") as f:
    f.write(reply + "\n")
print(result)
