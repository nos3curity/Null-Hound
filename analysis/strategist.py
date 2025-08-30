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

    def __init__(self, config: Optional[Dict[str, Any]] = None, debug: bool = False, session_id: Optional[str] = None, debug_logger=None):
        self.config = config or {}
        profile = _choose_profile(self.config)
        
        # Initialize or reuse debug logger
        self.debug_logger = debug_logger
        if debug and self.debug_logger is None:
            from analysis.debug_logger import DebugLogger
            self.debug_logger = DebugLogger(session_id or "strategist")
        
        self.profile = profile
        self.llm = UnifiedLLMClient(cfg=self.config, profile=profile, debug_logger=self.debug_logger)

    def plan_next(self, *, graphs_summary: str, completed: List[str], n: int = 5, 
                  hypotheses_summary: Optional[str] = None, coverage_summary: Optional[str] = None, 
                  ledger_summary: Optional[str] = None) -> List[Dict[str, Any]]:
        """Plan the next n investigations from comprehensive audit context.

        Returns a list of dicts compatible with downstream display and PlanStore.

        Prompt design notes (simple and commented for clarity):
        - Planning should start broad (aspect frames) then home in (suspicions) as evidence accumulates.
        - We do NOT encode a rigid slot ratio here – we instruct the model to maintain a sensible
          balance (e.g., ~60% aspects early, shifting to more suspicions later). This keeps logic simple
          and avoids brittle heuristics in code.
        - Each item must include a clear rationale; we ask the model to fold "why now" and
          "exit criteria" into the single 'reasoning' field, avoiding schema churn.
        """
        system = (
            "You are a senior smart-contract security auditor planning an audit roadmap.\n"
            "You have access to all graphs, annotations, previous findings, and coverage data.\n\n"
            "CRITICAL PLANNING PRINCIPLES:\n"
            "1. ALWAYS prioritize investigations with the highest chance of finding critical vulnerabilities\n"
            "2. Look for CONTRADICTIONS between assumptions and observations - these often reveal bugs\n"
            "3. Focus on areas where security controls intersect\n"
            "4. Reorganize priorities based on new findings - what seemed low-priority may become critical\n\n"
            "BREADTH→DEPTH MIX (guidance, not rigid):\n"
            "- Early batches: keep most items as broad ASPECT frames; reserve at least one slot for the strongest EVIDENCE-BASED SUSPICION.\n"
            "- As evidence accumulates (hypotheses/signals), increase SUSPICION items and revisit aspects only if coverage holes or contradictions persist.\n\n"
            "INVESTIGATION STRATEGY:\n"
            "- Start with HIGH-LEVEL security aspects and broad architectural patterns\n"
            "- As understanding accumulates, become progressively more specific\n"
            "- Use coverage metrics and completed investigations to gauge audit maturity\n"
            "- Early investigations should establish system understanding\n"
            "- Later investigations should target specific vulnerabilities based on evidence\n\n"
            "INVESTIGATION GUIDELINES:\n"
            "- Review ALL graph annotations (observations and assumptions) for contradictions or inconsistencies\n"
            "- Prioritize based on potential impact: critical > high > medium > low\n"
            "- Consider attack surface: external functions > internal functions > view functions\n"
            "- Look for common vulnerability patterns in unexplored areas\n"
            "- Build upon existing hypotheses - if we found issue X, check for related issue Y\n"
            "- Focus on uncovered nodes that handle value, permissions, or state changes\n"
            "- Avoid repeating completed investigations unless new evidence suggests revisiting\n"
            "- Provide exactly the requested number of items\n\n"
            "FOR EACH ITEM include in 'reasoning': (a) WHY NOW (signal/coverage need), and (b) EXIT CRITERIA (what evidence ends this thread).\n"
            "Use 'category' as 'aspect' or 'suspicion' and set 'expected_impact' realistically.\n"
        )

        completed_str = "\n".join(f"- {c}" for c in completed) if completed else "(none)"
        hypotheses_str = hypotheses_summary or "(no hypotheses formed yet)"
        coverage_str = coverage_summary or "(no coverage data)"
        ledger_str = ledger_summary or "(none)"
        
        # Calculate planning iteration count (this is passed as part of completed list)
        planning_iteration = len(completed) // n + 1 if n > 0 else 1
        
        user = (
            f"ALL GRAPHS WITH ANNOTATIONS:\n{graphs_summary}\n\n"
            f"CURRENT HYPOTHESES (vulnerabilities found):\n{hypotheses_str}\n\n"
            f"COMPLETED INVESTIGATIONS (with results):\n{completed_str}\n\n"
            f"COVERAGE STATUS:\n{coverage_str}\n\n"
            f"AUDIT PROGRESS:\n"
            f"- Planning iteration: {planning_iteration}\n"
            f"- Investigations completed: {len(completed)}\n\n"
            f"Plan the top {n} NEW investigations balancing breadth (ASPECTS) and depth (SUSPICIONS).\n"
            f"Favor ~60% ASPECTS early; shift toward SUSPICIONS as evidence accumulates.\n\n"
            f"PRIORITIZATION CRITERIA (in order):\n"
            f"1. Contradictions between assumptions and observations in the graphs\n"
            f"2. High-risk areas not yet covered\n"
            f"3. Patterns suggested by existing findings (if we found X, check for Y)\n"
            f"4. Complex interactions between multiple components\n"
            f"5. Areas with suspicious observations or questionable assumptions\n\n"
            f"For each investigation, include WHY NOW and EXIT CRITERIA in 'reasoning'."
        )

        # Allow fine-grained reasoning control for planning step
        plan_effort = None
        try:
            mdl_cfg = (self.config or {}).get('models', {}).get(self.profile, {})
            plan_effort = mdl_cfg.get('plan_reasoning_effort')
        except Exception:
            plan_effort = None
        plan: PlanBatch = self.llm.parse(system=system, user=user, schema=PlanBatch, reasoning_effort=plan_effort)
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
            "You are a deep-thinking senior smart-contract security auditor.\n"
            "Your job is to: (1) think deeply about the active investigation aspect,\n"
            "(2) uncover real, non-trivial vulnerabilities as clear hypotheses, and (3) advise the Scout on next steps.\n\n"
            "CRITICAL: Base your analysis on the investigation goal and the exploration/history shown in the context.\n"
            "Be precise, avoid generic statements, and only propose hypotheses you can justify from the provided context.\n"
            "If you are highly confident there are no vulnerabilities in scope, say so.\n"
        )
        user = (
            "CONTEXT (includes === INVESTIGATION GOAL === and compressed history):\n" + context + "\n\n"
            "OUTPUT INSTRUCTIONS:\n"
            "1) HYPOTHESES (one per line, exactly this pipe-separated format):\n"
            "   Title | Type | Root Cause | Attack Vector | Affected Nodes | Affected Code | Severity | Confidence | Reasoning\n"
            "   - severity: critical|high|medium|low; confidence: high|medium|low\n"
            "   - Keep Title concise and actionable.\n"
            "   - Affected Code should reference concrete functions/files if possible.\n\n"
            "2) GUIDANCE (next steps for the Scout):\n"
            "   - Provide 2–5 concrete next actions to gather evidence or rule in/out the above hypotheses.\n"
            "   - Reference specific nodes/functions/files the Scout should load or analyze next.\n\n"
            "3) If NO credible hypothesis is found, include a line: NO_HYPOTHESES: true (still provide GUIDANCE).\n"
        )

        # Save deep_think prompts to debug files if debug logger is available
        if self.debug_logger:
            try:
                from pathlib import Path
                from datetime import datetime
                # Prefer debug logger's output_dir for consistency; fallback to CWD/.hound_debug
                base_dir = getattr(self.debug_logger, 'output_dir', None)
                if not base_dir:
                    base_dir = Path.cwd() / '.hound_debug'
                base_path = Path(base_dir)
                # Create deep_think directory scoped by session id
                session_dir = base_path / str(getattr(self.debug_logger, 'session_id', 'strategist')) / 'deep_think_prompts'
                session_dir.mkdir(parents=True, exist_ok=True)
                # Timestamped filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                prompt_path = session_dir / f'deep_think_{timestamp}.txt'
                # Write full prompt for reproduction
                with open(prompt_path, 'w') as f:
                    f.write("=" * 80 + "\n")
                    f.write("DEEP THINK PROMPT\n")
                    f.write(f"Generated at: {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("SYSTEM PROMPT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(system)
                    f.write("\n\n")
                    f.write("USER PROMPT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(user)
                    f.write("\n\n")
                    f.write("=" * 80 + "\n")
                    f.write("NOTE: Combine the system and user prompts when testing manually.\n")
                    f.write("=" * 80 + "\n")
                # Optionally log an event into the HTML debug log
                try:
                    if hasattr(self.debug_logger, 'log_event'):
                        self.debug_logger.log_event('DeepThink Prompt Saved', str(prompt_path))
                except Exception:
                    pass
            except Exception:
                # Never fail deep_think on debug save issues
                pass

        try:
            # Allow fine-grained reasoning control for hypothesis step
            hyp_effort = None
            try:
                mdl_cfg = (self.config or {}).get('models', {}).get(self.profile, {})
                hyp_effort = mdl_cfg.get('hypothesize_reasoning_effort')
            except Exception:
                hyp_effort = None
            resp = self.llm.raw(system=system, user=user, reasoning_effort=hyp_effort)
            # Keep raw strategist output for downstream CLI display
            try:
                self.last_raw = resp
            except Exception:
                pass
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
