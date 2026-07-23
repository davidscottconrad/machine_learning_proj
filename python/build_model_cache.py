#!/usr/bin/env python3
"""Pre-train and cache the RF/LogReg models + historical window snapshots to
python/model_cache.joblib, so the first `predict.py` call doesn't pay the ~3s
training cost. `get_models()` also rebuilds this automatically the first time
any source CSV's mtime changes -- this script just lets you pre-warm it ahead
of time (e.g. right after cloning, or after updating a data file).
"""
from __future__ import annotations

import time

from predictor.model import get_models


def main() -> None:
    start = time.monotonic()
    models = get_models()
    elapsed = time.monotonic() - start
    print(f"Model cache built/verified in {elapsed:.2f}s.")
    print("Metrics:", models.metrics)


if __name__ == "__main__":
    main()
