"""Settings for the check in tracker and the model contest."""

TIMEZONE = "America/Detroit"

# Submissions created after this moment are not accepted. (ISO format, Detroit time.)
# None = no deadline yet. Example once you know it: "2026-12-09T17:00:00-05:00"
DEADLINE = None

# Shown in replies and on the tracker. Keep "TBD" until the date is set, then use e.g. "Dec 9".
CONTEST_DAY = "TBD"

# A project needs at least this many entrants to give out a prize.
MIN_ENTRANTS_FOR_PRIZE = 3

# GitHub usernames whose check-ins and submissions are ignored (project managers testing the bot).
IGNORE_LOGINS = ["MintMena"]

# One entry per project. "rows" must match the hidden features file.
PROJECTS = {
    "Poker Hand": {
        "key": "poker",
        "rows": 2480,
        "columns": ["prediction"],
        "task": "multiclass",
        "metric": "macro-F1",
        "tiebreak": "accuracy",
        "higher_is_better": True,
    },
    "Real Estate": {
        "key": "real_estate",
        "rows": 41,
        "columns": ["prediction"],
        "task": "regression",
        "metric": "RMSE",
        "tiebreak": "MAE",
        "higher_is_better": False,
    },
    "Power Plant": {
        "key": "power_plant",
        "rows": 957,
        "columns": ["prediction"],
        "task": "regression",
        "metric": "RMSE",
        "tiebreak": "MAE",
        "higher_is_better": False,
    },
    "Heart Failure": {
        "key": "heart_failure",
        "rows": 30,
        "columns": ["prediction", "probability"],
        "task": "binary",
        "metric": "ROC-AUC",
        "tiebreak": "MCC",
        "higher_is_better": True,
    },
}

WEEKS = [f"Week {i:02d}" for i in range(1, 7)]
CONTEST_ANSWERS = ["Yes", "Maybe", "No"]
