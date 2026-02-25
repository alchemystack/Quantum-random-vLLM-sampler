"""Data types for the token selection subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Result of CDF-based token selection.

    Attributes:
        token_id: Vocabulary index of the selected token.
        token_rank: Rank among probability-sorted candidates (0 = most probable).
        token_prob: Probability of the selected token after filtering.
        num_candidates: Number of tokens surviving top-k and top-p filtering.
        diagnostics: Additional info (CDF details, filtering stats).
    """

    token_id: int
    token_rank: int
    token_prob: float
    num_candidates: int
    diagnostics: dict[str, Any]
