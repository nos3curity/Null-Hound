"""LLM-assisted hypothesis deduplication utilities.

Uses a lightweight model profile to compare a new hypothesis against
existing ones in small batches to avoid duplicates.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from llm.unified_client import UnifiedLLMClient


class _DupResult(BaseModel):
    """Structured response from the dedup model for a batch."""
    duplicates: list[str] = Field(default_factory=list, description="List of existing hypothesis IDs that are semantically duplicates")
    rationale: str | None = None


def _get_lightweight_client(cfg: dict[str, Any], debug_logger=None) -> UnifiedLLMClient | None:
    """Try to create a lightweight client; fall back to scout/agent if missing."""
    try:
        return UnifiedLLMClient(cfg=cfg, profile="lightweight", debug_logger=debug_logger)
    except Exception:
        # Fallbacks to stay resilient in older configs
        for profile in ("scout", "agent"):
            try:
                return UnifiedLLMClient(cfg=cfg, profile=profile, debug_logger=debug_logger)
            except Exception:
                continue
    return None


def check_duplicates_llm(
    *,
    cfg: dict[str, Any],
    new_hypothesis: dict[str, Any],
    existing_batch: Iterable[dict[str, Any]],
    debug_logger=None,
) -> set[str]:
    """Return set of existing hypothesis IDs that are duplicates of the new one.

    Input "new_hypothesis" should include: title, description, vulnerability_type, node_refs (list[str]).
    Each item in "existing_batch" should include: id, title, description, vulnerability_type, node_refs.
    """
    client = _get_lightweight_client(cfg, debug_logger)
    if not client:
        return set()

    # Minimal, clear prompt for robust judging
    system = (
        "You are a fast, precise deduplication helper for security hypotheses.\n"
        "Two items are duplicates iff they describe essentially the same problem:\n"
        "- Same or highly similar root cause and code path(s)\n"
        "- Same contract/function(s)/state interaction, even with different wording\n"
        "- Ignore minor phrasing differences and synonyms\n"
        "They are NOT duplicates if the root cause, scope, or affected code paths are materially different.\n"
        "Return strict JSON only."
    )

    def _fmt_h(h: dict[str, Any]) -> str:
        nid = ",".join(h.get("node_refs") or [])
        return (
            f"id={h.get('id','unknown')}\n"
            f"title={h.get('title') or h.get('description','')}\n"
            f"type={h.get('vulnerability_type','unknown')}\n"
            f"nodes={nid}\n"
            f"desc={h.get('description','')}\n"
        )

    existing_lines = []
    for i, h in enumerate(existing_batch, 1):
        existing_lines.append(f"[{i}]\n" + _fmt_h(h))

    user = (
        "NEW HYPOTHESIS:\n" + _fmt_h(new_hypothesis) + "\n" +
        "EXISTING HYPOTHESES (CANDIDATE DUPLICATES):\n" + "\n".join(existing_lines) + "\n\n" +
        "Task: Identify duplicates by id.\n"
        "Output strict JSON: {\"duplicates\":[\"hyp_abc123\",...],\"rationale\":\"short reason\"}"
    )

    try:
        result = client.parse(system=system, user=user, schema=_DupResult)
        ids = set([str(x).strip() for x in (result.duplicates or []) if str(x).strip()])
        return ids
    except Exception:
        # Never block on dedup failures
        return set()

