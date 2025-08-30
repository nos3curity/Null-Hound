"""QA pre-filter – alias for finalization prefiltering under new naming."""

from typing import Dict, List, Tuple

from .finalization_prefilter import pre_filter_hypotheses, apply_filter_decisions

__all__ = [
    'pre_filter_hypotheses',
    'apply_filter_decisions',
]

