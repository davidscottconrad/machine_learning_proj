"""Tool schemas + local executors exposed to the Claude agent.

`target_year` (which tournament is being predicted) is bound server-side from
the request, not supplied by the model — it's deterministic context, not
something the LLM needs to reason about.
"""
from __future__ import annotations

from .model import TrainedModels

TOOLS = [
    {
        "name": "get_ml_prediction",
        "description": (
            "Run the trained Random Forest / Logistic Regression models for a specific matchup. "
            "Team strength (Elo, rolling form, win-rate) is computed from all international matches "
            "(friendlies, qualifiers, continental championships, World Cups) in the 2 years before "
            "the target tournament. Returns win/draw/loss probabilities from both models."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Team listed first (2026 squad name)"},
                "away_team": {"type": "string", "description": "Team listed second (2026 squad name)"},
            },
            "required": ["home_team", "away_team"],
        },
    },
    {
        "name": "get_team_stats",
        "description": (
            "Look up a single team's engineered strength features for the target tournament's "
            "2-year pre-tournament window: Elo rating, recent form (avg goal diff, last 5 matches), "
            "and win-rate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"team": {"type": "string", "description": "2026 squad team name"}},
            "required": ["team"],
        },
    },
    {
        "name": "get_head_to_head",
        "description": "List every historical international match (any competition, all-time) played directly between two teams.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string"},
                "team_b": {"type": "string"},
            },
            "required": ["team_a", "team_b"],
        },
    },
    {
        "name": "submit_prediction",
        "description": (
            "Submit your final prediction for this match. Call this exactly once, as your last action, "
            "after you've gathered enough information from the other tools. Probabilities must sum to 1.0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "winner": {
                    "type": "string",
                    "description": "'home', 'away', or 'draw' — the single most likely outcome",
                    "enum": ["home", "away", "draw"],
                },
                "home_win_prob": {"type": "number"},
                "draw_prob": {"type": "number"},
                "away_win_prob": {"type": "number"},
                "reasoning": {
                    "type": "string",
                    "description": "2-4 sentence explanation citing the specific stats/tools you used",
                },
            },
            "required": ["winner", "home_win_prob", "draw_prob", "away_win_prob", "reasoning"],
        },
    },
]


def execute_tool(name: str, tool_input: dict, models: TrainedModels, target_year: int) -> dict:
    if name == "get_ml_prediction":
        return models.predict_proba(tool_input["home_team"], tool_input["away_team"], target_year)
    if name == "get_team_stats":
        return models.get_team_stats(tool_input["team"], target_year)
    if name == "get_head_to_head":
        return models.get_head_to_head(tool_input["team_a"], tool_input["team_b"])
    raise ValueError(f"Unknown tool: {name}")
