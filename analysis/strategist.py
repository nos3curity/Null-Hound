"""Strategist (senior) planner.

Phase 2 introduces a minimal Strategist that can compose planning prompts and
return structured plan items. The CLI will wire this in a later step.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..llm.unified_client import UnifiedLLMClient


class PlanItemSchema(BaseModel):
    goal: str = Field(description="Investigation goal or question")
    focus_areas: List[str] = Field(default_factory=list)
    priority: int = Field(ge=1, le=10, description="1-10, 10 = highest")
    reasoning: str = Field(default="", description="Why this is promising")
    category: str = Field(default="aspect", description="aspect | suspicion")
    expected_impact: str = Field(default="medium", description="high | medium | low")


class PlanBatch(BaseModel):
    investigations: List[PlanItemSchema] = Field(default_factory=list)


def _choose_profile(cfg: Dict[str, Any]) -> str:
    # Prefer explicit strategist, then guidance, then agent as last resort
    try:
        models = cfg.get("models", {})
        if "strategist" in models:
            return "strategist"
        if "guidance" in models:
            return "guidance"
        return "agent"
    except Exception:
        return "agent"


class Strategist:
    """Senior planning agent."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        profile = _choose_profile(self.config)
        self.llm = UnifiedLLMClient(cfg=self.config, profile=profile)

    def plan_next(self, *, graphs_summary: str, completed: List[str], n: int = 5) -> List[Dict[str, Any]]:
        """Plan the next n investigations from a graphs + history summary.

        Returns a list of dicts compatible with downstream display and PlanStore.
        """
        system = (
            "You are a senior smart-contract security auditor planning an audit roadmap.\n"
            "Plan the next investigations based on the system architecture graph.\n\n"
            "GUIDELINES:\n"
            "- Prefer HIGH-LEVEL aspects to review unless the graph suggests a specific risk.\n"
            "- Avoid repeating completed items.\n"
            "- Provide exactly the requested number of items if possible.\n"
        )

        completed_str = "\n".join(f"- {c}" for c in completed) if completed else "(none)"
        user = (
            f"SYSTEM GRAPH SUMMARY:\n{graphs_summary}\n\n"
            f"ALREADY COMPLETED:\n{completed_str}\n\n"
            f"Plan the top {n} NEW investigations."
        )

        plan: PlanBatch = self.llm.parse(system=system, user=user, schema=PlanBatch)
        items = []
        for it in plan.investigations[:n]:
            items.append({
                "goal": it.goal,
                "focus_areas": it.focus_areas,
                "priority": it.priority,
                "reasoning": it.reasoning,
                "category": it.category,
                "expected_impact": it.expected_impact,
            })
        return items

    def revise_after(self, last_result: Dict[str, Any]) -> None:
        return None

__all__ = ["Strategist", "PlanItemSchema", "PlanBatch"]
