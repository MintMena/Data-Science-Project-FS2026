"""Streamlit leaderboard and participation dashboard for the model contest.

Run locally:  streamlit run leaderboard/app.py
Deploy free:  share.streamlit.io -> New app -> this repo, main file leaderboard/app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

# Change these two lines if the repo or branch name changes.
REPO = "MintMena/Data-Science-Project-FS2026"
BRANCH = "main"

RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ["Poker Hand", "Real Estate", "Power Plant", "Heart Failure"]
WEEKS = [f"Week {i:02d}" for i in range(1, 7)]


@st.cache_data(ttl=120)
def load(path):
    """Read a CSV from GitHub (always fresh), or from this folder if that fails."""
    try:
        return pd.read_csv(f"{RAW}/{path}")
    except Exception:
        local = ROOT / path
        return pd.read_csv(local) if local.exists() else pd.DataFrame()


st.set_page_config(page_title="DSC Model Contest", page_icon="🏆", layout="wide")
st.title("MSU Data Science Club: Guided Projects")
st.caption("Fall 2026 · Poker Hand, Real Estate, Power Plant, Heart Failure")

board, people = st.tabs(["🏆 Leaderboard", "📊 Participation"])

with board:
    results = load("leaderboard/results.csv")
    if results.empty:
        st.info("The leaderboard is revealed at the model contest meeting. Check back then!")
    else:
        cols = st.columns(2)
        for i, project in enumerate(PROJECTS):
            p = results[results["project"] == project]
            with cols[i % 2]:
                st.subheader(project)
                if p.empty:
                    st.write("No valid submissions.")
                    continue
                metric, tie = p["metric"].iloc[0], p["tiebreak_metric"].iloc[0]
                show = p[["rank", "name", "score", "tiebreak"]].rename(
                    columns={"rank": "Rank", "name": "Name", "score": metric, "tiebreak": tie})
                st.dataframe(show, hide_index=True, width="stretch",
                             column_config={metric: st.column_config.NumberColumn(format="%.4f"),
                                            tie: st.column_config.NumberColumn(format="%.4f")})

with people:
    checkins = load("tracking/checkins.csv")
    subs = load("tracking/submissions.csv")
    if checkins.empty:
        st.info("No check-ins yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Students checked in", checkins[["project", "login"]].drop_duplicates().shape[0])
        c2.metric("Check-ins", len(checkins))
        valid = subs[subs["valid"].astype(str).str.lower() == "true"] if not subs.empty else subs
        c3.metric("Valid contest files", 0 if valid.empty else valid[["project", "login"]].drop_duplicates().shape[0])

        st.subheader("Students per week")
        weekly = (checkins.drop_duplicates(["project", "week", "login"])
                  .pivot_table(index="week", columns="project", values="login", aggfunc="count")
                  .reindex(index=WEEKS, columns=PROJECTS).fillna(0).astype(int))
        st.bar_chart(weekly, stack=False)

        st.subheader("Joining the model contest?")
        latest = checkins.sort_values("issue").groupby(["project", "login"]).tail(1)
        rsvp = (latest.pivot_table(index="project", columns="contest", values="login", aggfunc="count")
                .reindex(index=PROJECTS, columns=["Yes", "Maybe", "No"]).fillna(0).astype(int))
        st.dataframe(rsvp, width="stretch")
