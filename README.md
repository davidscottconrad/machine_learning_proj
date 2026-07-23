# World Cup 2026 Bracket Predictor

A Next.js bracket UI backed by a Claude tool-use agent (Anthropic Python SDK) that predicts
knockout-stage match outcomes using a Random Forest + Logistic Regression model built on top of
the Elo / rolling-form / win-rate feature engineering from `World_Cup_Predictor.ipynb`.

## Running it

```bash
# Python backend (one-time setup)
cd python
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux
cp .env.example .env   # then fill in ANTHROPIC_API_KEY

# Frontend
cd ../frontend
npm install
npm run dev
```

Open http://localhost:3000. Each "Simulate" click has the Next.js API route
(`app/api/predict/route.ts`) spawn `python/predict.py` as a subprocess, which loads the ML
models, calls the Claude agent (`python/predictor/agent.py`), and returns a structured
prediction.

**Cost:** ~$0.01-0.04 per match on `claude-opus-4-8` (2-6 tool-use turns — Claude gathers evidence
via `get_ml_prediction`/`get_team_stats`/`get_head_to_head`, then submits its own probability
estimate). The response includes Claude's own call alongside the raw, unmodified Random Forest and
Logistic Regression numbers, so all three opinions are visible side by side. A full 31-match 2026
bracket costs roughly $1.

**Speed:** training the Random Forest + Logistic Regression (loading ~49k international matches,
building a 2-year window snapshot for all 23 historical World Cups) takes a few seconds — wasted
work when every prediction is a fresh subprocess. `get_models()` caches the trained result to
`python/model_cache.joblib`, keyed on the source CSVs' mtimes, so a cache hit just deserializes
instead of retraining. Run `python build_model_cache.py` once to pre-warm it (or just let the first
prediction populate it); it rebuilds automatically if any source CSV changes.

## How team strength is computed

Team strength (Elo, rolling last-5-match form, win-rate) is **not** built from World Cup matches
alone. It's built from **`wc_data/wc_data/international_results.csv`** — every FIFA-recognized
men's international match (friendlies, qualifiers, continental championships, and World Cups),
~49,000 matches from 1872 to the present, sourced from the public
[martj42/international_results](https://github.com/martj42/international_results) dataset.

For any given target tournament, only matches in the **2 years immediately before that
tournament's own start date** are used — a fresh Elo walk starting at the base rating each window,
so:

- A 2026 prediction sees each team's form from mid-2024 through early-June 2026 — real qualifying
  and friendly results, not decades of World Cup history.
- A 2022 backtest only sees 2020-2022 form. A 2018 backtest only sees 2016-2018 form. Etc. Each
  window is independent — a 2014 prediction has no idea 2018 or 2022 happened.
- The Random Forest / Logistic Regression classifiers themselves are trained the same way: every
  historical World Cup match's feature row is computed from *that tournament's own* 2-year
  pre-window snapshot (not a single cumulative walk), so training and inference use identical
  methodology.

**Why not just use World Cup matches?** The World Cup is quadrennial, so a 2-year window built
from World-Cup-only data would almost always be empty (the previous WC is always 4 years back).
Using all international matches means there's real, recent signal — qualifiers and friendlies —
for every window.

**A note on the source data:** `international_results.csv` is community-maintained and already
contains real 2026 World Cup results (it was pulled after the tournament concluded). Anything
dated on or after the 2026 World Cup's start (2026-06-11) was stripped from the file before it was
committed, so there's no possibility of the "prediction" for a 2026 match silently peeking at the
real outcome. The same date-boundary logic (strictly `< tournament start`) is what keeps every
backtest leakage-free — a 2014 prediction never sees 2014 results either, only what happened
before kickoff.

`matches_1930_2022.csv` (the original notebook's World Cup-only file) is still used for two
things: telling the code *when* each past World Cup started (so it can find the right 2-year
window), and providing the training labels (the actual W/D/L outcome of each historical match).

## Project layout

- `World_Cup_Predictor.ipynb` — the original notebook (WC-only Elo, kept as-is for reference).
- `python/predictor/data.py` — loads `international_results.csv`, builds a 2-year
  pre-tournament-window Elo/form/win-rate snapshot (`build_window_stats`) for any target year.
- `python/predictor/model.py` — trains RF + Logistic Regression on windowed features from every
  historical World Cup, and serves `predict_proba(home, away, target_year)`.
- `python/predictor/agent.py` + `tools.py` — the Claude tool-use agent: gathers evidence with
  `get_ml_prediction`/`get_team_stats`/`get_head_to_head`, then submits its own probability
  estimate + reasoning via `submit_prediction`. The raw RF/LogReg numbers from the last
  `get_ml_prediction` call are returned alongside Claude's own call, unmodified.
- `python/predict.py` — CLI entrypoint the frontend spawns as a subprocess.
- `python/build_model_cache.py` — pre-trains and caches the RF/LogReg models (see "Speed" above).
- `python/build_historical_brackets.py` — one-off script that extracted the real 2014/2018/2022
  knockout brackets (with actual results) from `matches_1930_2022.csv` into
  `frontend/lib/historical/*.json`; `2026.json` was hand-built the same way from a verified ESPN
  bracket scrape. Used for backtesting the model against real outcomes.
- `frontend/` — Next.js app: bracket UI with 2026/2022/2018/2014 tabs (all four are backtests,
  since the session's clock is after the real 2026 tournament). Only each bracket's first round
  (the real draw) is fixed; every later round is populated by whichever team the model actually
  predicts to advance — like a real bracket pick'em, a wrong early pick means the rest of that
  branch is hypothetical. The Results tab always shows the real matchup/winner at every round,
  independent of what the live view predicted, so backtest accuracy stays well-defined.
