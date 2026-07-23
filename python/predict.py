#!/usr/bin/env python3
"""CLI entrypoint invoked as a subprocess from the Next.js API route.

Usage: python predict.py '{"home_team": "Spain", "away_team": "Austria", "target_year": 2026}'
Prints a single JSON object to stdout; errors go to stderr with a non-zero exit code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from predictor.agent import predict_match  # noqa: E402  (must load .env first)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: predict.py '<json with home_team, away_team>'", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(sys.argv[1])
    home_team = payload["home_team"]
    away_team = payload["away_team"]
    target_year = payload["target_year"]

    result = predict_match(home_team, away_team, target_year)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
