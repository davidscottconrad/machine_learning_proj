"""
One-off script: extract the real Round of 16 -> Final knockout brackets for the last
three completed World Cups (2014, 2018, 2022) from wc_data/wc_data/matches_1930_2022.csv,
determine the actual winner of each match (penalties included), and reorder each round
so that adjacent pairs feed the next round (round[r+1][i] = winner(round[r][2i], round[r][2i+1])),
matching the convention the frontend bracket already uses.

Writes frontend/lib/historical/<year>.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "wc_data" / "wc_data" / "matches_1930_2022.csv"
OUT_DIR = ROOT / "frontend" / "lib" / "historical"

ROUND_ORDER = ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]
ROUND_LABELS = ["Round of 16", "Quarterfinals", "Semifinals", "Final"]


def winner_of(row) -> str:
    if row.home_score != row.away_score:
        return row.home_team if row.home_score > row.away_score else row.away_team
    # regulation draw -> penalties
    return row.home_team if row.home_penalty > row.away_penalty else row.away_team


def get_matches(m: pd.DataFrame, year: int, round_name: str) -> list[dict]:
    sub = m[(m.Year == year) & (m.Round == round_name)].sort_values("Date")
    out = []
    for row in sub.itertuples():
        out.append({"home": row.home_team, "away": row.away_team, "winner": winner_of(row)})
    return out


def reorder(prev_round: list[dict], next_round: list[dict]) -> list[dict]:
    by_winner = {m["winner"]: m for m in prev_round}
    ordered = []
    for nm in next_round:
        ordered.append(by_winner[nm["home"]])
        ordered.append(by_winner[nm["away"]])
    return ordered


def build_year(m: pd.DataFrame, year: int) -> list[list[dict]]:
    raw = {name: get_matches(m, year, name) for name in ROUND_ORDER}
    final = raw["Final"]
    sf = reorder(raw["Semi-finals"], final)
    qf = reorder(raw["Quarter-finals"], sf)
    r16 = reorder(raw["Round of 16"], qf)
    return [r16, qf, sf, final]


def main() -> None:
    m = pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="replace")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in (2014, 2018, 2022):
        rounds = build_year(m, year)
        assert [len(r) for r in rounds] == [8, 4, 2, 1], f"{year}: unexpected round sizes"
        out_path = OUT_DIR / f"{year}.json"
        out_path.write_text(json.dumps({"year": year, "roundLabels": ROUND_LABELS, "rounds": rounds}, indent=2))
        champion = rounds[-1][0]["winner"]
        print(f"{year}: wrote {out_path.relative_to(ROOT)} (champion: {champion})")


if __name__ == "__main__":
    main()
