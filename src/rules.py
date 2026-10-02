"""Business triage rules layered on top of the model score."""
from __future__ import annotations

from . import config


def triage(probability: float, threshold: float, v14: float, v4: float) -> tuple[str, str]:
    """Return (status, recommended action) for one transaction.

    status is one of "fraud", "review" or "legit".
    Rules follow the two-stage alert system: auto-block when v14 < -5 AND v4 > 5;
    manual review when either condition holds on its own.
    """
    low_v14 = v14 < config.RULE_V14_THRESHOLD
    high_v4 = v4 > config.RULE_V4_THRESHOLD

    if probability >= threshold or (low_v14 and high_v4):
        return "fraud", "Block the transaction and flag it for manual review."
    if low_v14 or high_v4:
        signal = "v14 < -5" if low_v14 else "v4 > 5"
        return "review", f"Model score is below the threshold, but rule signal ({signal}) fired. Queue for manual review."
    return "legit", "Transaction is safe to process."
