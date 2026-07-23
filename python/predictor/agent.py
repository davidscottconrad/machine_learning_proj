"""Anthropic Python SDK tool-use agent that predicts a single match outcome.

The agent can call get_ml_prediction / get_team_stats / get_head_to_head to gather
evidence, then must finish by calling submit_prediction exactly once with its own
probability estimate and reasoning. That forced final tool call is what's shown as
"Claude's" prediction — displayed alongside (not instead of) the raw Random Forest
and Logistic Regression numbers from the last get_ml_prediction call, so all three
opinions are visible side by side.
"""
from __future__ import annotations

import json
import os

from anthropic import Anthropic

from .model import TrainedModels, get_models
from .tools import TOOLS, execute_tool

MODEL_ID = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
MAX_TURNS = 6

SYSTEM_PROMPT = """\
You are a soccer match outcome predictor for the FIFA World Cup knockout stage.

You have tools that expose a Random Forest / Logistic Regression model (Elo + rolling-form + \
win-rate features, computed from all international matches in the 2 years before the target \
tournament), plus historical head-to-head results.

Use the tools to gather evidence, then call `submit_prediction` exactly once as your \
final action. Weigh the ML model's probabilities as your primary signal, adjusting only \
with clearly-stated reasoning (e.g. head-to-head history, notable form swings)."""

# Opus 4.8 pricing: $5.00 / MTok input, $25.00 / MTok output (see platform.claude.com/docs/pricing)
INPUT_PRICE_PER_MTOK = 5.00
OUTPUT_PRICE_PER_MTOK = 25.00


def _text_of(content) -> str:
    return "".join(b.text for b in content if b.type == "text")


def predict_match(home_team: str, away_team: str, target_year: int) -> dict:
    models: TrainedModels = get_models()
    client = Anthropic()

    messages = [
        {
            "role": "user",
            "content": (
                f"Predict the outcome of {home_team} (home) vs {away_team} (away) "
                f"at the {target_year} World Cup."
            ),
        }
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    last_ml_result: dict | None = None

    for turn in range(MAX_TURNS):
        force_final = turn == MAX_TURNS - 1
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "tool", "name": "submit_prediction"} if force_final else {"type": "auto"},
            messages=messages,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        submit = next((b for b in tool_uses if b.name == "submit_prediction"), None)
        if submit is not None:
            claude = dict(submit.input)
            cost = (
                total_input_tokens * INPUT_PRICE_PER_MTOK + total_output_tokens * OUTPUT_PRICE_PER_MTOK
            ) / 1_000_000
            # Knockout matches can't end in a draw, so even when the model's own
            # top pick is "draw" we still need a winner. In that case, defer to
            # whichever team was rated higher (Elo, computed the same way as the
            # ML features) at the time of the tournament, rather than the
            # home/away split of the tied prediction.
            probs = {
                "home": claude["home_win_prob"],
                "draw": claude["draw_prob"],
                "away": claude["away_win_prob"],
            }
            if max(probs, key=probs.get) == "draw":
                home_elo = models.get_team_stats(home_team, target_year)["elo"]
                away_elo = models.get_team_stats(away_team, target_year)["elo"]
                winner_side = "home" if home_elo >= away_elo else "away"
            else:
                winner_side = "home" if claude["home_win_prob"] >= claude["away_win_prob"] else "away"
            result = {
                "home_team": home_team,
                "away_team": away_team,
                "target_year": target_year,
                "winner": winner_side,
                "winner_team": home_team if winner_side == "home" else away_team,
                "claude": {
                    "home_win": claude["home_win_prob"],
                    "draw": claude["draw_prob"],
                    "away_win": claude["away_win_prob"],
                    "reasoning": claude["reasoning"],
                },
                "model": MODEL_ID,
                "usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "estimated_cost_usd": round(cost, 5),
                },
            }
            if last_ml_result is not None:
                result["random_forest"] = last_ml_result["random_forest"]
                result["logistic_regression"] = last_ml_result["logistic_regression"]
            return result

        if response.stop_reason != "tool_use" or not tool_uses:
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": "Call submit_prediction now with your final answer.",
                }
            )
            continue

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_uses:
            output = execute_tool(block.name, block.input, models, target_year)
            if block.name == "get_ml_prediction":
                last_ml_result = output
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": [{"type": "text", "text": json.dumps(output)}],
                }
            )
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError("Agent did not produce a prediction within the turn budget")
