"""PM ONLY. Pack the 4 hidden answer files into one GitHub secret.

Usage:  python pm_tools/make_answers_secret.py path/to/folder_with_answer_files
It writes secret.txt. Paste its contents into Settings -> Secrets and variables -> Actions
-> New repository secret, named HIDDEN_ANSWERS. Then delete secret.txt.
Never commit the answer files or secret.txt.
"""
import base64
import io
import sys
import zipfile
from pathlib import Path

FILES = ["poker_hidden_answers.csv", "real_estate_hidden_answers.csv",
         "power_plant_hidden_answers.csv", "heart_failure_hidden_answers.csv"]

folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
found = {name: next(folder.rglob(name), None) for name in FILES}
missing = [name for name, path in found.items() if path is None]
if missing:
    sys.exit("Missing: " + ", ".join(missing))

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for name, path in found.items():
        z.write(path, arcname=name)
secret = base64.b64encode(buf.getvalue()).decode()
if len(secret) > 48_000:
    sys.exit("Too big for a GitHub secret (48 KB limit).")
Path("secret.txt").write_text(secret)
print(f"Wrote secret.txt ({len(secret):,} characters). Paste it into a secret named HIDDEN_ANSWERS, then delete the file.")
