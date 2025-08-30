"""Strategist (senior) planner.

Phase 2 introduces a minimal Strategist that can compose planning prompts and
return structured plan items. The CLI will wire this in a later step.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from llm.unified_client import UnifiedLLMClient


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

    def plan_next(self, *, graphs_summary: str, completed: List[str], n: int = 5, coverage_summary: Optional[str] = None, ledger_summary: Optional[str] = None) -> List[Dict[str, Any]]:
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
        coverage_str = coverage_summary or "(none)"
        ledger_str = ledger_summary or "(none)"
        user = (
            f"SYSTEM GRAPH SUMMARY:\n{graphs_summary}\n\n"
            f"ALREADY COMPLETED:\n{completed_str}\n\n"
            f"RECENT COVERAGE (avoid redundant work unless new evidence warrants revisits):\n{coverage_str}\n\n"
            f"RECENT PROJECT FRAMES (informative only; DO NOT block if re-analysis is intentional):\n{ledger_str}\n\n"
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

    def deep_think(self, *, context: str) -> List[Dict[str, Any]]:
        """Perform senior deep analysis on the prepared context and emit hypothesis items.

        Returns a list of dicts with keys:
          description, details, vulnerability_type, severity, confidence, node_ids, reasoning
        """
        system = (
            "You are a deep-thinking smart-contract security auditor.\n"
            "Analyze the agent's exploration context to identify real, non-trivial vulnerabilities.\n"
            "Return hypotheses as ONE-LINE entries using '|' separators."
        )
        user = (
            "CONTEXT:\n" + context + "\n\n" +
            "HYPOTHESES (one per line):\n"
            "Title | Type | Root Cause | Attack Vector | Affected Nodes | Affected Code | Severity | Confidence | Reasoning\n"
            "Use: severity=(critical|high|medium|low), confidence=(high|medium|low)."
        )

        try:
            resp = self.llm.raw(system=system, user=user)
        except Exception:
            return []

        lines = [ln.strip() for ln in str(resp).splitlines() if ln.strip() and '|' in ln]
        items: List[Dict[str, Any]] = []
        for ln in lines:
            parts = [p.strip() for p in ln.split('|')]
            title = parts[0] if len(parts) > 0 else "Hypothesis"
            vuln_type = parts[1].lower() if len(parts) > 1 else "security_issue"
            root_cause = parts[2] if len(parts) > 2 else ""
            attack_vector = parts[3] if len(parts) > 3 else ""
            affected_nodes = parts[4] if len(parts) > 4 else ""
            affected_code = parts[5] if len(parts) > 5 else ""
            severity = parts[6].lower() if len(parts) > 6 else "medium"
            conf_word = parts[7].lower() if len(parts) > 7 else "medium"
            reasoning = parts[8] if len(parts) > 8 else ""

            confidence = 0.6
            if 'high' in conf_word:
                confidence = 0.9
            elif 'low' in conf_word:
                confidence = 0.4

            node_ids = [n.strip() for n in affected_nodes.split(',') if n.strip()] or ['system']
            details = (
                f"VULNERABILITY TYPE: {vuln_type}\n"
                f"ROOT CAUSE: {root_cause}\n"
                f"ATTACK VECTOR: {attack_vector}\n"
                f"AFFECTED NODES: {', '.join(node_ids)}\n"
                f"AFFECTED CODE: {affected_code}\n"
                f"SEVERITY: {severity}\n"
                f"REASONING: {reasoning}"
            )

            items.append({
                'description': title,
                'details': details,
                'vulnerability_type': vuln_type,
                'severity': severity,
                'confidence': confidence,
                'node_ids': node_ids,
                'reasoning': reasoning,
            })
        return items

__all__ = ["Strategist", "PlanItemSchema", "PlanBatch"]
