"""
Feature engineering for the World Cup predictor.

Team strength (Elo / rolling form / win-rate) is built from **all international
matches** (friendlies, qualifiers, continental championships, World Cups —
`international_results.csv`, ~49k matches, 1872-present) rather than World Cup
matches alone. For any given target tournament, only matches in the **2 years
immediately preceding that tournament's start** are used — a fresh Elo walk
starting at BASE_ELO each window, so a 2026 prediction never sees 2010 form,
and a 2018 backtest never sees 2022 results.

`matches_1930_2022.csv` (World Cup matches only) is used solely to know *when*
each past World Cup started, and to build the RF/LogReg training set.
"""
from __future__ import annotations

import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "wc_data" / "wc_data"

BASE_ELO, K = 1500.0, 40.0
WINDOW_YEARS = 2

# 2026-squad spelling -> spelling used in international_results.csv
NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def norm(s: str) -> str:
    return strip_accents(s).lower().strip()


def _result(home_score: int, away_score: int) -> str:
    return "H" if home_score > away_score else ("A" if home_score < away_score else "D")


@dataclass
class TeamStats:
    """A team-strength snapshot for one tournament's pre-window (elo/form/wr as of window end)."""

    elo_by_norm: dict
    form_by_norm: dict
    wr_by_norm: dict
    pts_by_norm: dict
    bridge_b0: float
    bridge_b1: float

    def resolve(self, name: str) -> str:
        for cand in (NAME_MAP.get(name, name), name):
            n = norm(cand)
            if n in self.elo_by_norm or n in self.pts_by_norm:
                return n
        return norm(name)

    def feats(self, name: str) -> tuple[float, float, float]:
        key = self.resolve(name)
        if key in self.elo_by_norm:
            return (
                self.elo_by_norm[key],
                self.form_by_norm.get(key, 0.0),
                self.wr_by_norm.get(key, 0.5),
            )
        if key in self.pts_by_norm:  # no matches in this window -> bridge from FIFA ranking points
            return self.bridge_b0 + self.bridge_b1 * self.pts_by_norm[key], 0.0, 0.5
        return 1400.0, 0.0, 0.5  # last-resort weak prior


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8", encoding_errors="replace")


def load_international_results() -> pd.DataFrame:
    df = _load_csv("international_results")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_fifa_points() -> dict:
    rank = _load_csv("fifa_ranking_2026-06-08")
    pts = {norm(t): p for t, p in zip(rank.team, rank.points)}
    # This file spells it "Cabo Verde"; international_results.csv spells it "Cape Verde".
    if "cabo verde" in pts:
        pts.setdefault("cape verde", pts["cabo verde"])
    return pts


def tournament_start_date(wc_matches: pd.DataFrame, year: int) -> pd.Timestamp:
    """First match date of a given past World Cup, from matches_1930_2022.csv."""
    return pd.to_datetime(wc_matches.loc[wc_matches.Year == year, "Date"]).min()


def resolve_name(name: str) -> str:
    return norm(NAME_MAP.get(name, name))


def head_to_head(intl: pd.DataFrame, team_a: str, team_b: str) -> list[dict]:
    """Every historical international match (any competition, all-time) between two teams."""
    a, b = resolve_name(team_a), resolve_name(team_b)
    home_norm = intl.home_team.map(norm)
    away_norm = intl.away_team.map(norm)
    mask = ((home_norm == a) & (away_norm == b)) | ((home_norm == b) & (away_norm == a))
    sub = intl[mask].dropna(subset=["home_score", "away_score"]).sort_values("date")
    return [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_score": int(row.home_score),
            "away_score": int(row.away_score),
            "tournament": row.tournament,
        }
        for row in sub.itertuples()
    ]


def build_window_stats(intl: pd.DataFrame, pts_by_norm: dict, window_end: pd.Timestamp) -> TeamStats:
    """
    Elo / rolling form / win-rate computed only from international matches in
    [window_end - WINDOW_YEARS, window_end) — a fresh walk each call, everyone
    starting at BASE_ELO. This is the team-strength snapshot going into a
    tournament that starts at `window_end`.
    """
    window_start = window_end - pd.DateOffset(years=WINDOW_YEARS)
    m = intl[(intl.date >= window_start) & (intl.date < window_end)]
    m = m.dropna(subset=["home_score", "away_score"])

    elo = defaultdict(lambda: BASE_ELO)
    last5 = defaultdict(lambda: deque(maxlen=5))
    wins = defaultdict(float)
    played = defaultdict(int)

    for row in m.itertuples():
        h, a = row.home_team, row.away_team
        hs, as_ = int(row.home_score), int(row.away_score)
        res = _result(hs, as_)

        eh, ea = elo[h], elo[a]
        exp_h = 1.0 / (1.0 + 10 ** ((ea - eh) / 400.0))
        s_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
        gd_mult = np.log(abs(hs - as_) + 1) + 1
        delta = K * gd_mult * (s_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

        gd = hs - as_
        last5[h].append(gd)
        last5[a].append(-gd)
        played[h] += 1
        played[a] += 1
        if res == "H":
            wins[h] += 1
        elif res == "A":
            wins[a] += 1
        else:
            wins[h] += 0.5
            wins[a] += 0.5

    elo_by_norm = {norm(k): v for k, v in elo.items()}
    form_by_norm = {norm(t): (np.mean(v) if v else 0.0) for t, v in last5.items()}
    wr_by_norm = {norm(t): (wins[t] / played[t] if played[t] else 0.5) for t in played}

    both = [(pts_by_norm[n], elo_by_norm[n]) for n in pts_by_norm if n in elo_by_norm]
    if len(both) >= 2:
        pts_arr, elo_arr = np.array(both).T
        b1, b0 = np.polyfit(pts_arr, elo_arr, 1)
    else:
        b0, b1 = BASE_ELO, 0.0

    return TeamStats(
        elo_by_norm=elo_by_norm,
        form_by_norm=form_by_norm,
        wr_by_norm=wr_by_norm,
        pts_by_norm=pts_by_norm,
        bridge_b0=float(b0),
        bridge_b1=float(b1),
    )
