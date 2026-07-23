"""
Model training for the World Cup predictor.

Random Forest + Logistic Regression, trained on elo_diff / form_diff / wr_diff —
same 3 features and hyperparameters as the notebook. The difference from the
notebook: those features now come from each tournament's own 2-year
pre-tournament window over *all* international matches (see data.py), not a
single running Elo walked across World Cup matches only. Every World Cup match
in the training set uses its own tournament's window snapshot, so a 1998 match
is scored on team strength as of 1998, not on decades of hindsight.

Training (loading ~49k international matches + fitting RF/LogReg) takes a few
seconds, which is wasted work when every `predict.py` invocation is a fresh
subprocess. `get_models()` caches the fully-trained result to
`python/model_cache.joblib`, keyed on the source CSVs' mtimes, so a cache hit
just deserializes the models instead of retraining. Run `build_model_cache.py`
to pre-warm the cache; it also rebuilds itself automatically the first time
any source CSV changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    TeamStats,
    build_window_stats,
    head_to_head,
    load_fifa_points,
    load_international_results,
    tournament_start_date,
)

RANDOM_STATE = 42
FEATURES = ["elo_diff", "form_diff", "wr_diff"]
LABELS = {"A": 0, "D": 1, "H": 2}  # team-2 win < draw < team-1 win

DATA_DIR = Path(__file__).resolve().parents[2] / "wc_data" / "wc_data"
CACHE_PATH = Path(__file__).resolve().parents[1] / "model_cache.joblib"
_SOURCE_FILES = [
    DATA_DIR / "international_results.csv",
    DATA_DIR / "matches_1930_2022.csv",
    DATA_DIR / "fifa_ranking_2026-06-08.csv",
    DATA_DIR / "schedule_2026.csv",
]


def _result(home_score: int, away_score: int) -> str:
    return "H" if home_score > away_score else ("A" if home_score < away_score else "D")


@dataclass
class TrainedModels:
    intl: pd.DataFrame
    pts_by_norm: dict
    wc_matches: pd.DataFrame
    logreg: object
    rf: object
    metrics: dict
    _window_cache: dict = field(default_factory=dict)

    def stats_for_target(self, target_year: int) -> TeamStats:
        if target_year in self._window_cache:
            return self._window_cache[target_year]

        if (self.wc_matches.Year == target_year).any():
            window_end = tournament_start_date(self.wc_matches, target_year)
        else:
            # Future tournament (2026): use the schedule file's earliest kickoff.
            sched = pd.read_csv(DATA_DIR / "schedule_2026.csv")
            window_end = pd.to_datetime(sched.Date).min()

        stats = build_window_stats(self.intl, self.pts_by_norm, window_end)
        self._window_cache[target_year] = stats
        return stats

    def predict_proba(self, home_team: str, away_team: str, target_year: int) -> dict:
        stats = self.stats_for_target(target_year)
        eh, fh, wh = stats.feats(home_team)
        ea, fa, wa = stats.feats(away_team)
        x = pd.DataFrame([[eh - ea, fh - fa, wh - wa]], columns=FEATURES)

        def probs(model):
            p = model.predict_proba(x)[0]  # [A, D, H]
            return {"home_win": float(p[2]), "draw": float(p[1]), "away_win": float(p[0])}

        return {
            "home_team": home_team,
            "away_team": away_team,
            "target_year": target_year,
            "features": {
                "home_elo": round(eh, 1),
                "away_elo": round(ea, 1),
                "home_form": round(fh, 3),
                "away_form": round(fa, 3),
                "home_winrate": round(wh, 3),
                "away_winrate": round(wa, 3),
            },
            "random_forest": probs(self.rf),
            "logistic_regression": probs(self.logreg),
            "model_metrics": self.metrics,
        }

    def get_team_stats(self, team: str, target_year: int) -> dict:
        stats = self.stats_for_target(target_year)
        elo, form, wr = stats.feats(team)
        return {"team": team, "elo": round(elo, 1), "form": round(form, 3), "winrate": round(wr, 3)}

    def get_head_to_head(self, team_a: str, team_b: str) -> dict:
        matches = head_to_head(self.intl, team_a, team_b)
        return {"matches": matches, "count": len(matches)}


def _expand_row(stats: TeamStats, home: str, away: str, res: str) -> list[tuple[list[float], int]]:
    eh, fh, wh = stats.feats(home)
    ea, fa, wa = stats.feats(away)
    straight = ([eh - ea, fh - fa, wh - wa], LABELS[res])
    flipped = {"H": "A", "D": "D", "A": "H"}[res]
    mirror = ([ea - eh, fa - fh, wa - wh], LABELS[flipped])
    return [straight, mirror]


def _source_signature() -> tuple:
    return tuple(f.stat().st_mtime_ns for f in _SOURCE_FILES)


def _load_cached() -> TrainedModels | None:
    if not CACHE_PATH.exists():
        return None
    try:
        payload = joblib.load(CACHE_PATH)
        if payload.get("signature") != _source_signature():
            return None  # a source CSV changed since the cache was built
        return payload["models"]
    except Exception:
        return None  # corrupt/incompatible cache (e.g. sklearn version bump) -> rebuild


def _save_cache(models: TrainedModels) -> None:
    joblib.dump({"signature": _source_signature(), "models": models}, CACHE_PATH)


def _train_fresh() -> TrainedModels:
    intl = load_international_results()
    pts_by_norm = load_fifa_points()

    wc_matches = pd.read_csv(DATA_DIR / "matches_1930_2022.csv", encoding="utf-8", encoding_errors="replace")
    wc_matches = wc_matches.dropna(subset=["home_score", "away_score"]).copy()
    wc_matches["home_score"] = wc_matches["home_score"].astype(int)
    wc_matches["away_score"] = wc_matches["away_score"].astype(int)

    # Build one window snapshot per historical World Cup, and training rows from it.
    rows: list[tuple[list[float], int]] = []
    window_cache: dict[int, TeamStats] = {}
    for year in sorted(wc_matches.Year.unique()):
        window_end = tournament_start_date(wc_matches, int(year))
        stats = build_window_stats(intl, pts_by_norm, window_end)
        window_cache[int(year)] = stats
        year_matches = wc_matches[wc_matches.Year == year]
        for row in year_matches.itertuples():
            res = _result(row.home_score, row.away_score)
            rows.extend(_expand_row(stats, row.home_team, row.away_team, res))

    X = pd.DataFrame([r[0] for r in rows], columns=FEATURES)
    y = pd.Series([r[1] for r in rows])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    metrics = {}
    for name, model in (("random_forest", rf), ("logistic_regression", logreg)):
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "macro_f1": round(f1_score(y_test, pred, average="macro"), 3),
            "log_loss": round(log_loss(y_test, proba), 3),
        }
    majority = y_train.value_counts().idxmax()
    metrics["majority_baseline_accuracy"] = round((y_test == majority).mean(), 3)

    return TrainedModels(
        intl=intl,
        pts_by_norm=pts_by_norm,
        wc_matches=wc_matches,
        logreg=logreg,
        rf=rf,
        metrics=metrics,
        _window_cache=window_cache,
    )


_CACHE: TrainedModels | None = None


def get_models() -> TrainedModels:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    _CACHE = _load_cached()
    if _CACHE is not None:
        return _CACHE

    _CACHE = _train_fresh()
    _save_cache(_CACHE)
    return _CACHE
