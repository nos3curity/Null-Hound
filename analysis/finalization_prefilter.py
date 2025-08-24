"""
Pre-filtering module for hypothesis finalization.
Uses LLM to intelligently filter out admin/governance/trivial issues in batch.
"""

import json
from typing import Dict, List, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def pre_filter_hypotheses(
    hypotheses: Dict[str, Dict], 
    threshold: float = 0.5, 
    llm=None, 
    debug: bool = False
) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, Dict]]]:
    """
    Pre-filter hypotheses using LLM to identify admin/governance/trivial issues.
    Processes all hypotheses in a single batch for efficiency.
    
    Args:
        hypotheses: Dictionary of hypothesis_id -> hypothesis_dict
        threshold: Minimum confidence threshold for consideration
        llm: LLM client for intelligent filtering
        debug: Enable debug output
    
    Returns:
        Tuple of (candidates_to_review, filtered_out)
        - candidates_to_review: List of (hypothesis_id, hypothesis_dict) tuples
        - filtered_out: List of (hypothesis_id, hypothesis_dict, reason) tuples
    """
    filtered_out = []
    candidates = []
    
    # Get eligible hypotheses
    eligible = []
    for hid, hyp in hypotheses.items():
        if hyp.get("status") not in ["confirmed", "rejected"] and hyp.get("confidence", 0) >= threshold:
            eligible.append((hid, hyp))
    
    if not eligible:
        return [], []
    
    if not llm:
        # Fallback to simple threshold filtering if no LLM
        return eligible, []
    
    # Build batch prompt with all hypotheses
    hypothesis_list = []
    for idx, (hid, hyp) in enumerate(eligible):
        hypothesis_list.append({
            "index": idx,
            "id": hid,
            "title": hyp.get('title', 'Unknown'),
            "type": hyp.get('vulnerability_type', 'unknown'),
            "severity": hyp.get('severity', 'unknown'),
            "description": hyp.get('description', 'No description')[:500]  # Limit length
        })
    
    batch_prompt = f"""Analyze this list of security hypotheses and determine which should be FILTERED OUT.

FILTER OUT if ANY of these apply:
1. Requires admin/owner/governance privileges to exploit (e.g., "admin can", "owner can", "governance can")
2. Is a deployment/initialization issue that only affects setup phase
3. Is a trivial issue (gas optimization, naming convention, typo, comment)
4. Requires governance majority/voting to exploit
5. Is about centralization risk or trust assumptions
6. Is a best practice suggestion without real security impact
7. Requires social engineering or off-chain coordination

KEEP if:
- Can be exploited by any user or attacker without special privileges
- Is a real vulnerability with security impact
- Could cause fund loss, DoS, or protocol malfunction
- Has a clear attack vector that doesn't require special roles
- Is exploitable in normal protocol operation

Hypotheses to review:
{json.dumps(hypothesis_list, indent=2)}

Respond with a JSON object containing a "decisions" array. Each item should have:
- "index": the hypothesis index from the list
- "decision": "FILTER" or "KEEP"
- "reason": Brief explanation (max 50 chars) if filtered

Example response:
{{
    "decisions": [
        {{"index": 0, "decision": "FILTER", "reason": "Requires admin role"}},
        {{"index": 1, "decision": "KEEP"}},
        {{"index": 2, "decision": "FILTER", "reason": "Trivial gas optimization"}}
    ]
}}"""
    
    # Process with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(
            f"Pre-filtering {len(eligible)} hypotheses with LLM (batch)...", 
            total=1
        )
        
        try:
            # Make batch LLM call
            if debug:
                console.print(f"[dim]Making batch LLM call for {len(eligible)} hypotheses...[/dim]")
            response_text = llm.raw(
                system="You are a security expert. Analyze these hypotheses and respond with valid JSON only.",
                user=batch_prompt
            )
            from utils.json_utils import extract_json_object
            response = extract_json_object(response_text)
            if not isinstance(response, dict):
                raise ValueError("Failed to parse JSON response from LLM")
            decisions_map = {}
            
            if "decisions" in response:
                for decision in response["decisions"]:
                    idx = decision.get("index")
                    if idx is not None and 0 <= idx < len(eligible):
                        decisions_map[idx] = {
                            "decision": decision.get("decision", "KEEP"),
                            "reason": decision.get("reason", "No reason provided")
                        }
            
            # Apply decisions
            for idx, (hid, hyp) in enumerate(eligible):
                decision_info = decisions_map.get(idx, {"decision": "KEEP"})
                
                if decision_info["decision"] == "FILTER":
                    reason = decision_info.get("reason", "Filtered by LLM")
                    filtered_out.append((hid, hyp, reason))
                    if debug:
                        title = hyp.get('title', 'Unknown')[:50]
                        progress.console.print(f"  [yellow]Filtered:[/yellow] {title}")
                        progress.console.print(f"    [dim]Reason: {reason}[/dim]")
                else:
                    candidates.append((hid, hyp))
                    if debug:
                        title = hyp.get('title', 'Unknown')[:50]
                        progress.console.print(f"  [green]Keeping:[/green] {title}")
            
            progress.advance(task)
            
        except Exception as e:
            if debug:
                progress.console.print(f"  [red]Error in batch filtering: {e}[/red]")
                progress.console.print("  [yellow]Falling back to keeping all hypotheses[/yellow]")
            # On error, keep all hypotheses to be safe
            candidates = eligible
            progress.advance(task)
    
    return candidates, filtered_out


def apply_filter_decisions(store, filtered_out: List[Tuple[str, Dict, str]], debug: bool = False):
    """
    Apply filtering decisions to the hypothesis store.
    
    Args:
        store: HypothesisStore instance
        filtered_out: List of (hypothesis_id, hypothesis_dict, reason) tuples
        debug: Enable debug output
    """
    if not filtered_out:
        return
    
    console.print(f"\n[cyan]Updating {len(filtered_out)} pre-filtered hypotheses...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Rejecting pre-filtered hypotheses...", total=len(filtered_out))
        
        for hid, hyp, reason in filtered_out:
            # Update confidence to 0 and add rejection reason
            store.adjust_confidence(hid, 0.0, f"Pre-filtered: {reason}")
            
            # Update status to rejected
            def update_status(data):
                if hid in data["hypotheses"]:
                    data["hypotheses"][hid]["status"] = "rejected"
                    data["hypotheses"][hid]["rejection_reason"] = f"Pre-filtered: {reason}"
                return data, True
            
            store.update_atomic(update_status)
            
            if debug:
                title = hyp.get('title', 'Unknown')[:50]
                progress.console.print(f"  [red]✗[/red] {title}")
            
            progress.advance(task)
