"""
Clean autonomous agent implementation - works like Claude Code.
No fuzzy parsing, no prescriptive flow, just autonomous decision-making.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
try:
    # Pydantic v2 style config
    from pydantic import ConfigDict  # type: ignore
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore

from .concurrent_knowledge import GraphStore


class AgentParameters(BaseModel):
    """Strict parameters schema for agent actions (no extra keys)."""
    # load_graph
    graph_name: Optional[str] = None
    # load_nodes
    node_ids: Optional[List[str]] = None
    # update_node
    node_id: Optional[str] = None
    observations: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    # form_hypothesis
    description: Optional[str] = None
    details: Optional[str] = None  # Added to support Gemini's response format
    vulnerability_type: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[str] = None
    reasoning: Optional[str] = None
    # update_hypothesis
    hypothesis_index: Optional[int] = None
    hypothesis_id: Optional[str] = None
    new_confidence: Optional[float] = None
    evidence: Optional[str] = None
    evidence_type: Optional[str] = None

    # Ensure OpenAI JSON schema has additionalProperties: false
    if ConfigDict is not None:  # Pydantic v2
        model_config = ConfigDict(extra='forbid')  # type: ignore
    else:  # Fallback for safety
        class Config:  # type: ignore
            extra = 'forbid'


class AgentDecision(BaseModel):
    """Structured decision from the agent."""
    action: str = Field(..., description="Action to take: load_graph, load_nodes, update_node, form_hypothesis, update_hypothesis, complete")
    reasoning: str = Field(..., description="Reasoning for this action")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters as shown in examples")
    
    # Pydantic v2 config for strict validation
    if ConfigDict is not None:  # Pydantic v2
        model_config = ConfigDict(extra='forbid')  # type: ignore
    else:  # Pydantic v1 fallback
        class Config:  # type: ignore
            extra = 'forbid'


class AutonomousAgent:
    """
    Security analysis agent that works autonomously.
    """
    
    def __init__(self, 
                 graphs_metadata_path: Path,
                 manifest_path: Path,
                 agent_id: str,
                 config: Optional[Dict] = None,
                 debug: bool = False,
                 session_id: Optional[str] = None):
        """Initialize the autonomous agent."""
        
        self.agent_id = agent_id
        self.manifest_path = manifest_path
        self.debug = debug
        self.session_id = session_id
        # Default hypothesis visibility; can be overridden by runner
        self.default_hypothesis_visibility = 'global'
        
        # Initialize debug logger if needed
        self.debug_logger = None
        if debug:
            from .debug_logger import DebugLogger
            self.debug_logger = DebugLogger(agent_id)
        
        # Initialize LLM client with proper config
        from llm.unified_client import UnifiedLLMClient
        
        # Use provided config or load defaults
        if config is None:
            from commands.graph import load_config
            config = load_config()
        
        # Save config for later utilities
        self.config = config

        # Check if there are platform/model overrides in the 'agent' profile
        # If so, use those instead of the 'scout' profile
        if (config and 'models' in config and 'agent' in config['models'] and 
            (config['models']['agent'].get('provider') or config['models']['agent'].get('model'))):
            # Use 'agent' profile when overrides are present
            profile_to_use = "agent"
        else:
            # Fall back to 'scout' profile (default behavior)
            profile_to_use = "scout"
        
        # Use the determined profile for agent operations
        self.llm = UnifiedLLMClient(
            cfg=config,
            profile=profile_to_use,
            debug_logger=self.debug_logger
        )
        
        # Initialize strategist model for deep thinking
        try:
            self.guidance_client = UnifiedLLMClient(
                cfg=config,
                profile="strategist",
                debug_logger=self.debug_logger
            )
        except Exception:
            # If strategist model not configured, fall back to scout model
            self.guidance_client = self.llm

        # Use the agent model itself for context compression
        # This ensures consistency and leverages the same model's understanding
        self.summarizer = self.llm  # Just use the agent's own LLM
        
        # Remember where graphs live and load metadata
        self.graphs_metadata_path = graphs_metadata_path
        self.available_graphs = self._load_graphs_metadata(graphs_metadata_path)
        
        # Initialize persistent hypothesis store (separate from graphs)
        # Store in the project directory for persistence
        from .concurrent_knowledge import HypothesisStore
        project_dir = graphs_metadata_path.parent.parent  # Go up to project root from graphs/
        hypothesis_path = project_dir / "hypotheses.json"
        self.hypothesis_store = HypothesisStore(hypothesis_path, agent_id=agent_id)
        
        # Agent's memory - what it has loaded and discovered
        self.loaded_data = {
            'system_graph': None,  # The always-visible system architecture graph
            'nodes': {},       # Loaded node data by ID
            'code': {},        # Code content by node ID
            'hypotheses': [],  # Formed hypotheses (kept for backward compatibility)
            'graphs': {},      # Additional loaded graphs by name
        }
        # Lazy card index
        self._card_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._file_to_cards: Dict[str, List[str]] = {}
        # Repo root to reconstruct card slices when needed
        self._repo_root: Optional[Path] = None
        try:
            with open(self.manifest_path / 'manifest.json') as _mf:
                _manifest = json.load(_mf)
                rp = _manifest.get('repo_path')
                if rp:
                    self._repo_root = Path(rp)
        except Exception:
            self._repo_root = None
        
        # AUTO-LOAD the system architecture graph (first graph, usually SystemArchitecture or SystemOverview)
        self._auto_load_system_graph()
        
        # Load existing hypotheses from persistent store
        self._load_existing_hypotheses()
        
        # Conversation history for context
        self.conversation_history = []
        # Compressed memory notes
        self.memory_notes: List[str] = []
        # High-level action log
        self.action_log: List[Dict[str, Any]] = []
        
        # Current investigation goal
        self.investigation_goal = ""
    
    def _load_graphs_metadata(self, metadata_path: Path) -> Dict:
        """Load metadata about available graphs."""
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        # Convert to expected format
        graphs = {}
        if 'graphs' in data:
            for name, path in data['graphs'].items():
                graphs[name] = {'path': path, 'name': name}
        
        return graphs
    
    def _auto_load_system_graph(self):
        """Automatically load the SystemOverview/SystemArchitecture graph at startup.
        Prefers file 'graph_SystemArchitecture.json' when present.
        """
        system_graph_name = None
        system_graph_path = None

        # 1) Exact key match 'SystemArchitecture'
        if 'SystemArchitecture' in self.available_graphs:
            system_graph_name = 'SystemArchitecture'
            system_graph_path = Path(self.available_graphs['SystemArchitecture']['path'])

        # 2) Heuristic match by name
        if not system_graph_path:
            for name in self.available_graphs.keys():
                lname = name.lower()
                if 'system' in lname and ('architecture' in lname or 'overview' in lname):
                    system_graph_name = name
                    system_graph_path = Path(self.available_graphs[name]['path'])
                    break

        # 3) Direct file presence: graph_SystemArchitecture.json next to metadata
        if not system_graph_path and hasattr(self, 'graphs_metadata_path') and self.graphs_metadata_path:
            candidate = Path(self.graphs_metadata_path).parent / 'graph_SystemArchitecture.json'
            if candidate.exists():
                system_graph_name = 'SystemArchitecture'
                system_graph_path = candidate

        # 4) Inspect graph files to find internal/display name of SystemArchitecture
        if not system_graph_path:
            for name, meta in self.available_graphs.items():
                try:
                    # Use concurrent-safe reload
                    gd = self._reload_graph(name)
                    if not gd:
                        continue
                    # Check common fields
                    internal = gd.get('internal_name') or gd.get('name') or ''
                    display = gd.get('name') or gd.get('metadata', {}).get('display_name') or ''
                    if str(internal) == 'SystemArchitecture' or str(display).strip().lower() in {'systemarchitecture', 'system architecture', 'systemoverview', 'system overview'}:
                        system_graph_name = name
                        system_graph_path = Path(meta['path'])
                        break
                except Exception:
                    continue

        # 5) Fallback to the first available
        if not system_graph_path and self.available_graphs:
            first_name = list(self.available_graphs.keys())[0]
            system_graph_name = first_name
            system_graph_path = Path(self.available_graphs[first_name]['path'])

        # Load selected graph
        if system_graph_name:
            try:
                # Use concurrent-safe reload
                graph_data = self._reload_graph(system_graph_name)
                if graph_data:
                    self.loaded_data['system_graph'] = {
                        'name': system_graph_name,
                        'data': graph_data
                    }

                    nodes = graph_data.get('nodes', [])
                    edges = graph_data.get('edges', [])
                    print(f"[*] Auto-loaded system graph: {system_graph_name} ({len(nodes)} nodes, {len(edges)} edges)")
            except Exception as e:
                print(f"[!] Failed to auto-load system graph: {e}")
    
    def _load_existing_hypotheses(self):
        """Load existing hypotheses from persistent store into memory."""
        try:
            # Clear existing loaded hypotheses to get fresh view
            self.loaded_data['hypotheses'] = []
            
            # Load from persistent store
            data = self.hypothesis_store._load_data()
            hypotheses = data.get("hypotheses", {})
            
            # Convert to memory format
            for hyp_id, hyp in hypotheses.items():
                self.loaded_data['hypotheses'].append({
                    'id': hyp_id,
                    'description': hyp.get('title', 'Unknown'),
                    'vulnerability_type': hyp.get('vulnerability_type', 'unknown'),
                    'confidence': hyp.get('confidence', 0.5),
                    'status': hyp.get('status', 'proposed'),
                    'node_ids': hyp.get('node_refs', []),
                    'evidence': [e.get('description') for e in hyp.get('evidence', [])]
                })
            
            # Only print on first load, not every refresh
            # if len(self.loaded_data['hypotheses']) > 0:
            #     print(f"[*] Loaded {len(self.loaded_data['hypotheses'])} existing hypotheses")
        except Exception as e:
            print(f"[!] Failed to load existing hypotheses: {e}")
    
    def _refresh_loaded_graphs(self):
        """Refresh loaded graphs from disk to see updates from other agents."""
        try:
            # Refresh system graph if loaded
            if self.loaded_data.get('system_graph'):
                graph_name = self.loaded_data['system_graph']['name']
                refreshed = self._reload_graph(graph_name)
                if refreshed:
                    self.loaded_data['system_graph']['data'] = refreshed
            
            # Refresh any additional loaded graphs
            for graph_name in list(self.loaded_data.get('graphs', {}).keys()):
                refreshed = self._reload_graph(graph_name)
                if refreshed:
                    self.loaded_data['graphs'][graph_name] = refreshed
        except Exception as e:
            print(f"[!] Failed to refresh graphs: {e}")
    
    def investigate(self, prompt: str, max_iterations: int = 20, 
                   progress_callback: Optional[callable] = None) -> Dict:
        """
        Main investigation method - agent works autonomously until complete.
        """
        self.investigation_goal = prompt
        self.conversation_history = [
            {'role': 'user', 'content': prompt}
        ]
        
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            if progress_callback:
                progress_callback({
                    'status': 'analyzing',
                    'iteration': iterations,
                    'message': 'Analyzing context and deciding next action'
                })
            
            try:
                # Build full context for agent
                context = self._build_context()
                
                # Get agent's decision using structured output
                decision = self._get_agent_decision(context)
                # Surface the agent's decision and reasoning to UI
                if progress_callback:
                    try:
                        progress_callback({
                            'status': 'decision',
                            'iteration': iterations,
                            'action': decision.action,
                            'reasoning': decision.reasoning,
                            'parameters': decision.parameters,  # Already a dict
                            'message': f"Decided to {decision.action}"
                        })
                    except Exception:
                        pass
                
                # Log the decision
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': f"Action: {decision.action}\nReasoning: {decision.reasoning}"
                })
                
                if progress_callback:
                    progress_callback({
                        'status': 'executing',
                        'iteration': iterations,
                        'message': f"Executing: {decision.action}"
                    })
                
                # Execute the decision
                result = self._execute_action(decision)
                # Surface result to UI (generic)
                if progress_callback:
                    try:
                        progress_callback({
                            'status': 'result',
                            'iteration': iterations,
                            'action': decision.action,
                            'result': result,
                            'message': result.get('summary') or f"{decision.action} -> {result.get('status', 'done')}"
                        })
                    except Exception:
                        pass
                
                # Log the result - use formatted display for readability
                # For successful graph/node loads, show the formatted display
                if result.get('status') == 'success':
                    if 'graph_display' in result:
                        # load_graph action - show formatted graph
                        content = f"SUCCESS: {result.get('summary', '')}\n{result['graph_display']}"
                    elif 'nodes_display' in result:
                        # load_nodes action - show formatted nodes with code
                        content = f"SUCCESS: {result.get('summary', '')}\n{result['nodes_display']}"
                    else:
                        # Other successful actions - show as JSON but more readable
                        content = json.dumps(result, indent=2)
                else:
                    # Errors and other statuses - show as JSON
                    content = json.dumps(result, indent=2)
                
                self.conversation_history.append({
                    'role': 'system',
                    'content': content
                })

                # Record action in action log (compact, only non-null params)
                try:
                    # Parameters is now a dict, filter out None values
                    params_obj = {k: v for k, v in decision.parameters.items() if v is not None}
                except Exception:
                    params_obj = {}
                self.action_log.append({
                    'action': decision.action,
                    'params': params_obj,
                    'result': result.get('summary') or result.get('status') or 'ok'
                })

                # Maybe compress history if near budget
                self._maybe_compress_history()
                
                # Check if complete
                if decision.action == 'complete':
                    if progress_callback:
                        progress_callback({
                            'status': 'complete',
                            'iteration': iterations,
                            'message': 'Investigation complete'
                        })
                    break
                
                # Update progress based on action
                if decision.action == 'form_hypothesis' and result.get('status') == 'success':
                    if progress_callback:
                        progress_callback({
                            'status': 'hypothesis_formed',
                            'iteration': iterations,
                            'message': f"Formed hypothesis: {decision.parameters.get('description', 'Unknown')}"
                        })
                elif decision.action == 'load_nodes' and result.get('status') == 'success':
                    if progress_callback:
                        progress_callback({
                            'status': 'code_loaded',
                            'iteration': iterations,
                            'message': result.get('summary', 'Loaded nodes')
                        })
                        
            except Exception as e:
                error_msg = f"Error in iteration {iterations}: {str(e)}"
                print(f"[!] {error_msg}")
                if self.debug:
                    traceback.print_exc()
                
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"ERROR: {error_msg}"
                })
        
        # Generate final report
        if progress_callback:
            progress_callback({
                'status': 'generating_report',
                'iteration': iterations,
                'message': 'Generating final report'
            })
        
        return self._generate_report(iterations)
    
    def _format_graph_for_display(self, graph_data: Dict, graph_name: str) -> List[str]:
        """Format a graph for compact display with observations/assumptions."""
        lines = []
        lines.append(f"\n--- Graph: {graph_name} ---")
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        lines.append(f"Total: {len(nodes)} nodes, {len(edges)} edges")
        lines.append("USE EXACT NODE IDS AS SHOWN BELOW - NO VARIATIONS!\n")
        
        # Show ALL nodes with top observations/assumptions
        if nodes:
            lines.append("AVAILABLE NODES (use these EXACT IDs with load_nodes):")
            lines.append("📊 Code size indicator: [S]=small (1-2 cards), [M]=medium (3-5), [L]=large (6+)")
            lines.append("PRIORITIZE [S] and [M] nodes! Only load [L] if absolutely necessary!\n")
            
            for node in nodes:
                node_id = node.get('id', 'unknown')
                node_label = node.get('label', node_id)
                node_type = node.get('type', 'unknown')
                
                # Count source_refs to estimate code size
                source_refs = node.get('source_refs', []) or []
                card_count = len(source_refs)
                
                # Size indicator
                if card_count == 0:
                    size_indicator = "[∅]"  # No code
                elif card_count <= 2:
                    size_indicator = "[S]"  # Small
                elif card_count <= 5:
                    size_indicator = "[M]"  # Medium
                else:
                    size_indicator = f"[L:{card_count}]"  # Large with count
                
                # Make node IDs stand out more
                lines.append(f"  {size_indicator} [{node_id}] → {node_label} ({node_type})")
                
                # Show annotations inline for maximum compactness
                observations = node.get('observations', [])
                assumptions = node.get('assumptions', [])
                
                # Collect annotation strings
                annots = []
                if observations:
                    sorted_obs = sorted(observations, 
                                      key=lambda x: x.get('confidence', 1.0) if isinstance(x, dict) else 1.0, 
                                      reverse=True)[:2]  # Reduced to 2 for compactness
                    obs_strs = []
                    for obs in sorted_obs:
                        if isinstance(obs, dict):
                            desc = obs.get('description', obs.get('content', str(obs)))
                            obs_strs.append(desc)
                        else:
                            obs_strs.append(str(obs))
                    if obs_strs:
                        annots.append(f"obs:{'; '.join(obs_strs)}")
                
                if assumptions:
                    sorted_assum = sorted(assumptions,
                                        key=lambda x: x.get('confidence', 0.5) if isinstance(x, dict) else 0.5,
                                        reverse=True)[:2]  # Reduced to 2 for compactness
                    assum_strs = []
                    for assum in sorted_assum:
                        if isinstance(assum, dict):
                            desc = assum.get('description', assum.get('content', str(assum)))
                            assum_strs.append(desc)
                        else:
                            assum_strs.append(str(assum))
                    if assum_strs:
                        annots.append(f"asm:{'; '.join(assum_strs)}")
                
                # Add annotations inline if present
                if annots:
                    lines.append(f"    [{' | '.join(annots)}]")
        
        # Show edge summary
        if edges:
            lines.append("\nEDGE TYPES:")
            edge_types = {}
            for edge in edges:
                edge_type = edge.get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            for edge_type, count in edge_types.items():
                lines.append(f"  • {edge_type}: {count} edges")
            
            # Show edges in compact form
            lines.append("\nEDGES (compact):")
            for edge in edges[:50]:  # Limit to first 50 edges for readability
                src = edge.get('source_id') or edge.get('source') or edge.get('src')
                dst = edge.get('target_id') or edge.get('target') or edge.get('dst')
                etype = edge.get('type', 'rel')
                # Build edge line with inline annotations
                edge_line = f"  {etype} {src}->{dst}"
                
                # Show edge observations/assumptions inline for compactness
                edge_obs = edge.get('observations', [])
                edge_assum = edge.get('assumptions', [])
                edge_annots = []
                if edge_obs:
                    obs_str = '; '.join(str(o) for o in edge_obs[:2])
                    edge_annots.append(f"obs:{obs_str}")
                if edge_assum:
                    assum_str = '; '.join(str(a) for a in edge_assum[:2])
                    edge_annots.append(f"asm:{assum_str}")
                
                if edge_annots:
                    edge_line += f" [{' | '.join(edge_annots)}]"
                lines.append(edge_line)
            
            if len(edges) > 50:
                lines.append(f"  ... and {len(edges) - 50} more edges")
        
        return lines

    def _build_context(self) -> str:
        """Build complete context for the agent to see.
        
        Permanent context includes:
        - Investigation goal
        - Available graphs list
        - System architecture graph (always visible)
        - Memory notes (compressed history)
        - Recent actions
        
        Temporary data (appears only in action history):
        - Loaded graphs (other than system graph)
        - Node details and source code
        """
        # Refresh hypotheses from store to see updates from other agents
        self._load_existing_hypotheses()
        
        # Reload graphs to see updates from other agents
        self._refresh_loaded_graphs()
        
        context_parts = []
        
        # Investigation goal
        context_parts.append(f"=== INVESTIGATION GOAL ===")
        context_parts.append(self.investigation_goal)
        context_parts.append("")
        
        # Available graphs (show all)
        context_parts.append("=== AVAILABLE GRAPHS ===")
        context_parts.append("Use EXACT graph names as shown below:")
        for name in self.available_graphs.keys():
            if self.loaded_data['system_graph'] and name == self.loaded_data['system_graph']['name']:
                context_parts.append(f"• {name} [SYSTEM - AUTO-LOADED, see nodes below]")
            else:
                context_parts.append(f"• {name}")
        context_parts.append("")

        # Compressed memory notes (if any)
        if self.memory_notes:
            context_parts.append("=== MEMORY (COMPRESSED HISTORY) ===")
            for note in self.memory_notes[-5:]:
                context_parts.append(f"• {note}")
            context_parts.append("")
        
        # System graph - ALWAYS VISIBLE with ALL NODES
        if self.loaded_data['system_graph']:
            context_parts.append("=== SYSTEM ARCHITECTURE (ALWAYS VISIBLE) ===")
            graph_name = self.loaded_data['system_graph']['name']
            graph_data = self.loaded_data['system_graph']['data']
            # Use unified formatting function
            context_parts.extend(self._format_graph_for_display(graph_data, graph_name))
        context_parts.append("")
        
        # Actions performed (recent) - summary only since full data is in RECENT ACTIONS
        if self.action_log:
            context_parts.append("=== ACTIONS PERFORMED (SUMMARY) ===")
            for entry in self.action_log[-10:]:
                act = entry.get('action','-')
                r = entry.get('result','')
                # Just show action and brief result summary
                if isinstance(r, str):
                    rs = r[:100]
                else:
                    rs = str(r)[:100]
                context_parts.append(f"- {act}: {rs}")
            context_parts.append("")

        # Current hypotheses (ALWAYS show, display clearly to prevent duplicates)
        context_parts.append("=== EXISTING HYPOTHESES (DO NOT DUPLICATE!) ===")
        if self.loaded_data['hypotheses']:
            # Group by vulnerability type to make duplicates obvious
            by_type = {}
            for hyp in self.loaded_data['hypotheses']:
                vtype = hyp.get('vulnerability_type', 'unknown')
                if vtype not in by_type:
                    by_type[vtype] = []
                by_type[vtype].append(hyp)
            
            for vtype, hyps in by_type.items():
                context_parts.append(f"\n{vtype.upper()}:")
                for hyp in hyps:
                    status = hyp.get('status', 'proposed')
                    conf = hyp['confidence']
                    # Show full title and affected nodes to prevent duplicates
                    if status == 'confirmed':
                        icon = '✓'
                    elif status == 'rejected':
                        icon = '✗'
                    elif status == 'supported':
                        icon = '+'
                    elif status == 'refuted':
                        icon = '-'
                    else:
                        icon = '?'
                    
                    nodes = hyp.get('node_ids', [])
                    nodes_str = ','.join(nodes[:3]) if nodes else 'unknown'
                    title = hyp.get('title', hyp['description'][:60])
                    
                    # Single compact line per hypothesis with clear info
                    context_parts.append(f"  [{icon}] {conf:.0%} @{nodes_str}: {title}")
        else:
            context_parts.append("None")
        context_parts.append("")
        
        # Recent actions (for context awareness)
        # Show ALL conversation history (compression will handle size limits)
        if len(self.conversation_history) > 1:
            context_parts.append("=== RECENT ACTIONS ===")
            # Show all entries - compression handles size management
            for entry in self.conversation_history:
                if entry['role'] == 'assistant':
                    context_parts.append(f"Action: {entry['content']}")
                elif entry['role'] == 'system':
                    # Include full result for system responses (contains graph/node data)
                    content = entry['content']
                    # Mark compressed entries clearly
                    if content.startswith('[MEMORY]'):
                        context_parts.append(f"===== COMPRESSED HISTORY =====\n{content}")
                    else:
                        context_parts.append(f"Result: {content}")
            context_parts.append("")
        
        return '\n'.join(context_parts)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using accurate tokenization when available."""
        try:
            from llm.tokenization import count_tokens
            return count_tokens(text, self.llm.provider_name, self.llm.model)
        except Exception as e:
            try:
                return max(1, len(text) // 4)
            except Exception:
                return 0

    def _maybe_compress_history(self):
        """Compress older conversation into memory notes when near context limit.
        
        Smart compression strategy:
        1. Compress when context reaches threshold (default 75% of max)
        2. Preserve recent actions (default last 5) in full detail
        3. Summarize older actions, focusing on:
           - Loaded graphs and their key findings
           - Formed hypotheses and confidence levels
           - Updated nodes with critical observations
           - Any errors or blockers encountered
        """
        # Get context management settings (global for both agent and guidance)
        context_cfg = (self.config or {}).get('context', {}) if isinstance(self.config, dict) else {}
        max_tokens = int(context_cfg.get('max_tokens', 128000))
        compression_threshold = float(context_cfg.get('compression_threshold', 0.75))
        keep_recent = int(context_cfg.get('keep_recent_actions', 5))
        
        # Calculate current context size using accurate tokenization when available
        try:
            current_context = self._build_context()
            current_tokens = self._count_tokens(current_context)
        except Exception:
            return
        
        # Check if compression is needed
        threshold_tokens = int(max_tokens * compression_threshold)
        if current_tokens < threshold_tokens:
            return  # Still have room
        
        # Log compression trigger
        print(f"\n[CONTEXT COMPRESSION] Triggered at {current_tokens}/{max_tokens} tokens ({current_tokens*100//max_tokens}% full)")
        print(f"[CONTEXT COMPRESSION] Compressing {len(self.conversation_history) - keep_recent} old entries, keeping {keep_recent} recent")
        
        if len(self.conversation_history) <= keep_recent + 1:
            print(f"[CONTEXT COMPRESSION] Not enough history to compress (only {len(self.conversation_history)} entries)")
            return  # Not enough history to compress
        
        # Split history into old (to compress) and recent (to keep)
        old_entries = self.conversation_history[:-keep_recent]
        recent_entries = self.conversation_history[-keep_recent:]
        
        # Intelligently extract key information from old entries
        graphs_loaded = set()
        nodes_analyzed = set()
        hypotheses_formed = []
        key_observations = []
        errors_encountered = []
        
        for entry in old_entries:
            role = entry.get('role', '')
            content = entry.get('content', '')
            
            if role == 'assistant':
                # Extract action type
                if 'load_graph' in content:
                    # Extract graph name from action
                    if 'Reasoning:' in content:
                        graphs_loaded.add(content.split('\n')[0].replace('Action: load_graph', '').strip())
            
            elif role == 'system':
                # Parse results
                if 'SUCCESS:' in content and 'Graph:' in content:
                    # Extract loaded graph info
                    for line in content.split('\n'):
                        if '--- Graph:' in line:
                            graph_name = line.split('Graph:')[1].split('---')[0].strip()
                            graphs_loaded.add(graph_name)
                        elif 'obs:' in line or 'assume:' in line:
                            # Capture important observations
                            key_observations.append(line.strip())
                            if len(key_observations) > 20:  # Keep only most important
                                key_observations = key_observations[-20:]
                
                elif 'nodes_display' in content or 'LOADED NODE DETAILS' in content:
                    # Extract analyzed nodes
                    for line in content.split('\n'):
                        if ' | ' in line and not line.startswith(' '):
                            node_id = line.split(' | ')[0].strip()
                            if node_id:
                                nodes_analyzed.add(node_id)
                
                elif '"status": "error"' in content or 'ERROR:' in content:
                    # Track errors
                    error_msg = content[:200]  # First 200 chars
                    if error_msg not in errors_encountered:
                        errors_encountered.append(error_msg)
                
                # Extract formed hypotheses
                if 'form_hypothesis' in content and 'success' in content:
                    hypotheses_formed.append("Hypothesis formed")
        
        # Build compressed summary
        summary_parts = []
        
        if graphs_loaded:
            summary_parts.append(f"Graphs analyzed: {', '.join(list(graphs_loaded)[:5])}")
        
        if nodes_analyzed:
            summary_parts.append(f"Nodes examined: {', '.join(list(nodes_analyzed)[:10])}")
        
        if hypotheses_formed:
            summary_parts.append(f"Hypotheses: {len(hypotheses_formed)} formed")
        
        if key_observations:
            # Include most recent key observations
            obs_summary = ' | '.join(key_observations[-5:])
            summary_parts.append(f"Key findings: {obs_summary[:300]}")
        
        if errors_encountered:
            summary_parts.append(f"Errors: {len(errors_encountered)} encountered")
        
        # Create final compressed note
        if summary_parts:
            summary_note = f"[Compressed {len(old_entries)} actions] " + " || ".join(summary_parts)
        else:
            # Fallback to simple compression
            summary_note = f"[Compressed {len(old_entries)} past actions]"
        
        # If we have a summarizer LLM, use it for better compression
        if self.summarizer and len(old_entries) > 10:
            try:
                # Prepare focused content for summarization
                important_content = []
                for entry in old_entries:
                    content = entry.get('content', '')
                    # Focus on results and key information
                    if 'SUCCESS:' in content or 'form_hypothesis' in content or 'obs:' in content:
                        important_content.append(content[:1000])  # First 1000 chars of important entries
                
                if important_content:
                    sys_p = """Summarize the key findings from this security audit into 5-8 bullets:
                    - Which graphs and nodes were analyzed
                    - What vulnerabilities or hypotheses were formed
                    - Critical observations or assumptions made
                    - Any errors or blockers encountered
                    Keep only the most important facts for continuing the audit."""
                    
                    user_p = '\n'.join(important_content[:30])  # Limit input size
                    resp = self.summarizer.raw(system=sys_p, user=user_p)
                    
                    if resp:
                        lines = [l.strip('-• ') for l in resp.splitlines() if l.strip()]
                        if lines:
                            summary_note = f"[AI-Compressed history] " + ' | '.join(lines[:8])
            except Exception:
                pass  # Keep the heuristic summary
        
        # Update memory and conversation history
        self.memory_notes.append(summary_note)
        
        # Keep only the compressed summary and recent entries
        self.conversation_history = [
            {'role': 'system', 'content': f"[MEMORY] {summary_note}"}
        ] + recent_entries
        
        # Also clear old entries from loaded_data to free memory (except system graph)
        # But keep the critical findings in memory notes
        graphs_cleared = 0
        if len(self.loaded_data.get('graphs', {})) > 3:
            # Keep only most recently loaded graphs
            graph_items = list(self.loaded_data['graphs'].items())
            graphs_cleared = len(self.loaded_data['graphs']) - 3
            self.loaded_data['graphs'] = dict(graph_items[-3:])
        
        nodes_cleared = 0
        if len(self.loaded_data.get('nodes', {})) > 10:
            # Keep only most recently loaded nodes
            node_items = list(self.loaded_data['nodes'].items())
            nodes_cleared = len(self.loaded_data['nodes']) - 10
            self.loaded_data['nodes'] = dict(node_items[-10:])
        
        # Log compression completion
        print(f"[CONTEXT COMPRESSION] Complete! Compressed {len(old_entries)} entries into memory note")
        print(f"[CONTEXT COMPRESSION] Summary: {summary_note[:200]}...")
        if graphs_cleared or nodes_cleared:
            print(f"[CONTEXT COMPRESSION] Cleared {graphs_cleared} old graphs and {nodes_cleared} old nodes from memory")
        
        # Recalculate tokens after compression using accurate counting
        try:
            new_context = self._build_context()
            new_tokens = self._count_tokens(new_context)
            print(f"[CONTEXT COMPRESSION] New context size: {new_tokens}/{max_tokens} tokens ({new_tokens*100//max_tokens}% full)")
        except Exception:
            pass  # Don't fail if we can't calculate new size
    
    def _get_agent_decision(self, context: str) -> AgentDecision:
        """
        Get agent's structured decision based on context.
        Uses provider-appropriate method for reliable parsing.
        """
        system_prompt = """You are an autonomous security investigation agent analyzing smart contracts.

YOUR CORE RESPONSIBILITY: You are the EXPLORER and CONTEXT BUILDER. Your primary job is to:
- Navigate and explore the graph structure to understand the system
- Load relevant code that implements the features being investigated  
- Build comprehensive context by examining multiple related components
- Prepare complete information for the deep think model to analyze

The deep think model (guidance) is a separate, expensive reasoning engine that performs vulnerability analysis.
It can ONLY analyze the context you prepare - if you don't load it, it can't analyze it!

Your task is to investigate the system and identify potential vulnerabilities. The system architecture graph is automatically loaded and visible. You can see all available graphs and which are loaded.

CRITICAL RULES FOR NODE AND GRAPH NAMES:
- ALWAYS use EXACT node IDs as shown in the graphs (in square brackets like [node_id])
- NEVER guess, modify, or create node names
- NEVER add prefixes like "func_" or "node_" unless they're already there
- If a node doesn't exist in the graph, DON'T try variations - it doesn't exist!
- Check the graph FIRST to see what nodes actually exist before requesting them

IMPORTANT DISTINCTION:
- Graph observations/assumptions: Facts about HOW the system works (invariants, behaviors, constraints)
- Hypotheses: Suspected SECURITY ISSUES or vulnerabilities
Never mix these - security concerns always go in hypotheses, not in graph updates.

WHEN ADDING OBSERVATIONS/ASSUMPTIONS:
Keep EXTREMELY SHORT - just essential facts, not full sentences:
- Good: "only owner", "checks balance", "emits Transfer", "immutable", "reentrancy guard"
- Bad: "This function can only be called by the owner of the contract"
- Bad: "The function checks that the balance is greater than zero before proceeding"

AVAILABLE ACTIONS - USE EXACT PARAMETERS AS SHOWN:

1. load_graph — Load an additional graph for analysis
   PARAMETERS: {"graph_name": "GraphName"}
   EXAMPLE: {"graph_name": "AuthorizationRoles"}
   EXAMPLE: {"graph_name": "DataFlowDiagram"}
   ONLY SEND: graph_name - NOTHING ELSE!

2. load_nodes — Load source code for specific nodes from a specific graph
   PARAMETERS: {"graph_name": "ExactGraphName", "node_ids": ["exact_node_id_from_brackets"]}
   REQUIRED: graph_name (string) AND node_ids (array)
   COPY THE EXACT NODE IDs from the square brackets [like_this] in the graph display
   
   LOADING STRATEGY:
   - PRIORITIZE nodes marked [S] (small) and [M] (medium) - these are targeted functions
   - AVOID nodes marked [L:n] (large) - these are entire contracts with many code blocks
   - Load specific functions (func_*) rather than entire contracts (contract_*)
   - If you must load a large node, explain WHY it's necessary
   
   CORRECT EXAMPLE: {"graph_name": "SystemArchitecture", "node_ids": ["func_AIToken_mint"]}
   WRONG EXAMPLE: {"graph_name": "System", "node_ids": ["contract_Agent"]} ← entire contract!
   
   The node IDs are shown in square brackets. Size indicators show code volume.

3. update_node — Add observations/assumptions about ONE node
   PARAMETERS: {"node_id": "node", "observations": [...], "assumptions": [...]}
   EXAMPLE: {"node_id": "ProxyAdmin", "observations": ["single admin", "no timelock"]}
   EXAMPLE: {"node_id": "func_transfer", "assumptions": ["checks balance"]}
   ONLY SEND: node_id (required), observations (optional), assumptions (optional)
   DO NOT SEND: Empty arrays [] - omit the field instead
   Keep observations/assumptions VERY SHORT (2-4 words each)

4. update_hypothesis — Update existing hypothesis with new evidence
   PARAMETERS: {"hypothesis_index": 0, "new_confidence": 0.5, "evidence": "..."}
   EXAMPLE: {"hypothesis_index": 0, "new_confidence": 0.9, "evidence": "Confirmed by analyzing implementation"}
   ONLY SEND: hypothesis_index, new_confidence, evidence - NOTHING ELSE!

5. deep_think — Analyze recent exploration for vulnerabilities (EXPENSIVE - use wisely!)
   PARAMETERS: {}
   EXAMPLE: {}
   Send empty object {} - NO PARAMETERS!
   
   CRITICAL PREREQUISITES - DO NOT CALL deep_think UNTIL:
   - You have loaded RELEVANT graphs for the investigation topic
   - You have loaded ACTUAL CODE (nodes) that implements the feature being investigated
   - You have made observations about the loaded code
   - You have a COMPLETE VIEW of the feature/subsystem being analyzed
   
   NEVER call deep_think:
   - At the start of investigation (no context loaded yet!)
   - After only loading graphs without loading any nodes
   - When you haven't explored the specific feature mentioned in the investigation
   
   ONLY call deep_think:
   - After loading and examining 5-10 relevant nodes minimum
   - When you have a complete understanding of a subsystem
   - When you need strategic guidance after thorough exploration
   
   Purpose: The deep think model performs expensive, thorough vulnerability analysis
   on the context YOU have prepared. It can only analyze what you've loaded!

6. complete — Finish the current investigation
   PARAMETERS: {}
   EXAMPLE: {}
   Send empty object {} - NO PARAMETERS!

YOUR PRIMARY ROLE - CONTEXT PREPARATION:
You are the NAVIGATOR and EXPLORER. Your job is to:
1. Navigate the graph structure to find relevant components
2. Load and examine code that implements the investigated feature
3. Build a complete understanding of how the system works
4. PREPARE comprehensive context for the deep think model to analyze

The deep think model is EXPENSIVE and can only analyze what YOU load!
Think of yourself as preparing a detailed case file for an expert analyst.

EXPLORATION STRATEGY:
1. UNDERSTAND the investigation goal - what feature/property are we examining?
2. LOAD relevant graphs that show this feature's structure
3. IDENTIFY nodes that implement this feature (check size indicators!)
4. LOAD the actual code (5-10+ nodes minimum) for these components
5. MAKE observations about how the code works
6. ONLY THEN call deep_think when you have a COMPLETE picture
7. Follow deep_think's guidance to explore related areas
8. Repeat: thorough exploration → deep_think → more exploration

SMART LOADING: Load func_* nodes (specific functions) rather than contract_* nodes (entire files)!
REMEMBER: deep_think can only analyze what you've loaded - incomplete context = incomplete analysis!

COMPLETION CRITERIA (WHEN TO CALL complete):
1. You have explored key areas AND deep_think has analyzed them for vulnerabilities, OR
2. Further exploration is unlikely to reveal new important information, OR
3. No promising exploration paths remain.

IMPORTANT: 
- Do NOT form hypotheses directly - that's deep_think's job
- NEVER call deep_think without loading substantial code context first (5-10+ nodes minimum)
- Deep_think is EXPENSIVE and analyzes YOUR discoveries - incomplete prep = wasted analysis
- Your role: EXPLORE thoroughly, LOAD relevant code, BUILD complete context
- Only call deep_think when you have a COMPLETE understanding of the investigated feature

EXPECTATIONS:
- Choose nodes at the most informative granularity (functions/storage) when available.
- Avoid loading entire contracts by default; only do so when specifically necessary.
- Be explicit in your reasoning about why each selected node advances the goal.

IMPORTANT JSON FORMATTING RULES:
- NEVER use null values - omit the field entirely if not needed
- Only include parameters that are REQUIRED for the specific action
- Do NOT include empty arrays [] or null - omit the field
- Each action has SPECIFIC required parameters - only include those

Return a JSON object with: action, reasoning, parameters"""
        
        user_prompt = f"""Current Context:

{context}

What is your next action? Respond ONLY with a valid JSON object in this exact format:
{{
  "action": "action_name",
  "reasoning": "why you are taking this action",
  "parameters": {{...action-specific parameters...}}
}}

DO NOT include any text before or after the JSON object."""
        
        # Use raw JSON output - works across all providers
        try:
            # First try raw call with JSON instruction
            response = self.llm.raw(
                system=system_prompt,
                user=user_prompt
            )
            
            # Parse the JSON response
            if response:
                # Clean response - remove markdown code blocks if present
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]  # Remove ```json
                if response.startswith('```'):
                    response = response[3:]  # Remove ```
                if response.endswith('```'):
                    response = response[:-3]  # Remove trailing ```
                response = response.strip()
                
                # Parse JSON
                data = json.loads(response)
                # Ensure parameters is a dict
                if 'parameters' not in data:
                    data['parameters'] = {}
                elif data['parameters'] is None:
                    data['parameters'] = {}
                return AgentDecision(**data)
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"[!] JSON parsing failed: {e}")
            # Fallback to more robust parsing
            if response:
                try:
                    from .parsing import parse_agent_decision_fallback
                    data = parse_agent_decision_fallback(response)
                    if isinstance(data, dict):
                        if 'parameters' not in data or data['parameters'] is None:
                            data['parameters'] = {}
                        return AgentDecision(**data)
                    # Try to extract action and reasoning manually
                    import re
                    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response)
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)
                    if action_match:
                        return AgentDecision(
                            action=action_match.group(1),
                            reasoning=reasoning_match.group(1) if reasoning_match else "Parsed from malformed JSON",
                            parameters={}
                        )
                except Exception as e2:
                    print(f"[!] Failed to parse response: {e2}")
            
            # Ultimate fallback - make a reasonable decision
            if not self.loaded_data['nodes']:
                # Look for critical nodes in the system graph
                critical_nodes = []
                if self.loaded_data['system_graph']:
                    graph_data = self.loaded_data['system_graph']['data']
                    for node in graph_data.get('nodes', [])[:5]:  # First 5 nodes
                        critical_nodes.append(node['id'])
                
                return AgentDecision(
                    action="load_nodes",
                    reasoning="Need to load node data to analyze code",
                    parameters={"node_ids": critical_nodes}
                )
            else:
                return AgentDecision(
                    action="complete",
                    reasoning="Unable to parse response, completing",
                    parameters={}
                )
    
    def _execute_action(self, decision: AgentDecision) -> Dict:
        """Execute the agent's decision."""
        action = decision.action
        params = decision.parameters  # Now a dict
        
        if action == 'deep_think':
            # Execute deep thinking analysis
            return self._deep_think()
        elif action == 'load_graph':
            # Extract graph_name
            graph_name = params.get('graph_name', '')
            if isinstance(graph_name, str):
                # Remove any JSON artifacts
                graph_name = graph_name.strip().rstrip("'}").rstrip('"').rstrip("'")
            return self._load_graph(graph_name)
        elif action == 'load_nodes':
            # Extract graph_name and node_ids
            graph_name = params.get('graph_name')
            node_ids = params.get('node_ids', [])
            return self._load_nodes(node_ids, graph_name)
        elif action == 'update_node':
            # Pass only relevant parameters
            clean_params = {
                'node_id': params.get('node_id')
            }
            if 'observations' in params:
                clean_params['observations'] = params['observations']
            if 'assumptions' in params:
                clean_params['assumptions'] = params['assumptions']
            return self._update_node(clean_params)
        elif action == 'form_hypothesis':
            # Pass hypothesis parameters
            return self._form_hypothesis(params)
        elif action == 'update_hypothesis':
            # Pass update parameters
            return self._update_hypothesis(params)
        elif action == 'complete':
            return {'status': 'complete', 'summary': 'Investigation complete'}
        else:
            return {'status': 'error', 'error': f'Unknown action: {action}'}
    
    def _load_graph(self, graph_name: str) -> Dict:
        """Load an additional knowledge graph.
        
        The graph data is returned in the action response (appearing in history)
        rather than being added to permanent context. Only the system graph
        remains permanently visible in context.
        """
        # Clean up graph name (remove quotes, trailing characters, JSON artifacts)
        if graph_name:
            # Remove common JSON/text artifacts
            graph_name = graph_name.strip().strip("'\"")
            # Split on common delimiters that shouldn't be in graph names
            for delimiter in ["'", '"', '}', ')', '\n', ' ']:
                if delimiter in graph_name:
                    graph_name = graph_name.split(delimiter)[0]
            graph_name = graph_name.strip()
        
        if not graph_name:
            return {'status': 'error', 'error': 'Graph name is required'}
        
        # Try to find the graph (case-insensitive match as fallback)
        if graph_name not in self.available_graphs:
            # Try case-insensitive match
            for available_name in self.available_graphs.keys():
                if available_name.lower() == graph_name.lower():
                    graph_name = available_name
                    break
            else:
                # Suggest similar names
                available = list(self.available_graphs.keys())
                return {'status': 'error', 'error': f'Graph not found: {graph_name}. Available: {", ".join(available)}'}
        
        # Don't reload the system graph
        if self.loaded_data['system_graph'] and graph_name == self.loaded_data['system_graph']['name']:
            return {
                'status': 'info',
                'summary': f'{graph_name} is already loaded as the system graph'
            }
        
        try:
            # Use concurrent-safe reload method
            graph_data = self._reload_graph(graph_name)
            if not graph_data:
                return {'status': 'error', 'error': f'Failed to load graph {graph_name}'}
            
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            # Store in loaded graphs so it appears in context
            self.loaded_data['graphs'][graph_name] = graph_data

            # Format the graph for display using unified function
            formatted_lines = self._format_graph_for_display(graph_data, graph_name)
            graph_display = '\n'.join(formatted_lines)
            
            return {
                'status': 'success',
                'summary': f'Loaded {graph_name}: {len(nodes)} nodes, {len(edges)} edges',
                'graph_display': graph_display
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    @property
    def cards(self):
        """Access card index, loading if needed."""
        self._ensure_card_index()
        return self._card_index or {}
    
    def _iterate_graphs(self):
        """Iterate over all loaded graphs (system + additional)."""
        # System graph - use actual graph name instead of 'system'
        if self.loaded_data.get('system_graph'):
            graph_name = self.loaded_data['system_graph']['name']
            yield graph_name, self.loaded_data['system_graph']['data']
        
        # Additional loaded graphs
        for name, data in self.loaded_data.get('graphs', {}).items():
            yield name, data
    
    def _ensure_card_index(self):
        """Load and cache cards.jsonl as an index by ID."""
        if self._card_index is not None:
            return
        from .cards import load_card_index
        idx, file_map = load_card_index(self.graphs_metadata_path, self.manifest_path)
        self._card_index = idx
        self._file_to_cards.update(file_map)

    def _extract_card_content(self, card: Dict[str, Any]) -> str:
        """Get best-available content from a card record."""
        from .cards import extract_card_content
        return extract_card_content(card, self._repo_root)

    # Note: _iter_graphs was a duplicate of _iterate_graphs and has been removed.

    def _load_nodes(self, node_ids: List[str], graph_name: Optional[str] = None) -> Dict:
        """Load complete node data with associated source code.
        
        Node details and code are returned in the action response (appearing in history)
        rather than being added to permanent context. This allows the agent to explore
        nodes without permanently filling the context window.
        
        Args:
            node_ids: List of node IDs to load
            graph_name: REQUIRED - The specific graph to load nodes from
        """
        if not graph_name:
            return {
                'status': 'error',
                'error': 'graph_name is REQUIRED. Please specify which graph contains the nodes you want to load.'
            }
            
        if not node_ids:
            return {
                'status': 'error',
                'error': 'No node IDs specified. Please specify which nodes to load from the graph.'
            }
        
        self._ensure_card_index()
        
        # First check if the graph exists and is loaded
        graph_data = None
        graph_edges = []
        
        # Check system graph
        if self.loaded_data.get('system_graph') and self.loaded_data['system_graph']['name'] == graph_name:
            graph_data = self.loaded_data['system_graph']['data']
        # Check loaded graphs
        elif graph_name in (self.loaded_data.get('graphs') or {}):
            graph_data = self.loaded_data['graphs'][graph_name]
        else:
            # Graph not loaded - provide helpful error
            available = []
            if self.loaded_data.get('system_graph'):
                available.append(self.loaded_data['system_graph']['name'])
            available.extend(self.loaded_data.get('graphs', {}).keys())
            return {
                'status': 'error',
                'error': f'Graph "{graph_name}" is not loaded. Available graphs: {", ".join(available) if available else "none"}. Use load_graph first if needed.'
            }
        
        # Build index for ONLY the specified graph
        node_by_id: Dict[str, Dict[str, Any]] = {}
        graph_edges = graph_data.get('edges', [])
        
        for n in graph_data.get('nodes', []):
            nid = n.get('id')
            if nid:
                node_by_id[nid] = n

        not_found: List[str] = []
        loaded_nodes = []

        for req_id in node_ids:
            # EXACT match only - no fuzzy matching
            ndata = node_by_id.get(req_id)
            
            if not ndata:
                not_found.append(req_id)
                continue
            
            chosen_id = req_id
            
            # Track large nodes for warning in display, but don't print here
            source_refs = ndata.get('source_refs', []) or []
            # Will show warning in display_lines instead of printing

            # Collect evidence cards from node and its incident edges
            # We already have graph_edges from the specified graph
            card_ids: List[str] = []
            node_refs = ndata.get('source_refs', []) or ndata.get('refs', []) or []
            if isinstance(node_refs, list):
                card_ids.extend([str(x) for x in node_refs])
            
            # Debug logging removed - too noisy
            for e in graph_edges:
                src = e.get('source_id') or e.get('source') or e.get('src')
                dst = e.get('target_id') or e.get('target') or e.get('dst')
                if src == chosen_id or dst == chosen_id:
                    evid = e.get('evidence', []) or e.get('source_refs', []) or []
                    if isinstance(evid, list):
                        card_ids.extend([str(x) for x in evid])

            # Dedup
            seen = set()
            base_ids = []
            for cid in card_ids:
                if cid and cid not in seen:
                    seen.add(cid)
                    base_ids.append(cid)

            # Resolve cards ordered by relpath + char_start
            node_cards: List[Dict[str, Any]] = []
            ordered = []
            for cid in base_ids:
                c = self._card_index.get(cid)
                if c:
                    ordered.append(c)
                # Card not found - continue silently
                pass
            
            # Debug logging removed
            
            ordered.sort(key=lambda x: (x.get('relpath') or '', x.get('char_start') or 0))
            for c in ordered:
                cid = c.get('id')
                content = self._extract_card_content(c)
                
                # Debug logging removed
                
                node_cards.append({
                    'card_id': cid,
                    'type': c.get('type', 'code'),
                    'content': content,
                    'metadata': {
                        k: c.get(k) for k in (
                            'relpath','char_start','char_end','line_start','line_end'
                        ) if k in c
                    }
                })
                # Track card coverage
                try:
                    if self.coverage_index and cid:
                        self.coverage_index.touch_card(str(cid))
                except Exception:
                    pass

            # NO FALLBACK - if node has no explicit source_refs, it has no code
            # This prevents loading entire files when agent requests non-existent nodes

            node_copy = ndata.copy()
            if node_cards:
                node_copy['cards'] = node_cards
                self.loaded_data['code'][chosen_id] = '\n\n'.join(c['content'] for c in node_cards)
            self.loaded_data['nodes'][chosen_id] = node_copy
            loaded_nodes.append(chosen_id)
            try:
                if self.coverage_index:
                    self.coverage_index.touch_node(chosen_id)
            except Exception:
                pass

        # Count only the nodes from this request
        current_request_nodes = loaded_nodes
        current_loaded_count = len(loaded_nodes)
        current_code_count = len([nid for nid in loaded_nodes if 'cards' in self.loaded_data['nodes'][nid]])
        
        # Total across all previous loads (for context)
        total_loaded_count = len(self.loaded_data['nodes'])
        
        # Format loaded nodes for display
        display_lines = []
        display_lines.append(f"\n=== LOADED NODE DETAILS FROM {graph_name} ===")
        display_lines.append(f"This request: {current_loaded_count} nodes ({current_code_count} with code)")
        
        # Check if any large nodes were loaded
        large_nodes_loaded = []
        for nid in loaded_nodes:
            node_data = self.loaded_data['nodes'][nid]
            if 'cards' in node_data and len(node_data['cards']) > 6:
                large_nodes_loaded.append(f"{nid}({len(node_data['cards'])} blocks)")
        
        if large_nodes_loaded:
            display_lines.append(f"⚠️ WARNING: Loaded LARGE nodes: {', '.join(large_nodes_loaded)}")
            display_lines.append("Consider loading specific functions instead of entire contracts!")
        
        if total_loaded_count > current_loaded_count:
            display_lines.append(f"Total cached: {total_loaded_count} nodes\n")
        else:
            display_lines.append("")
        
        for node_id in current_request_nodes:
            node_data = self.loaded_data['nodes'][node_id]
            node_type = node_data.get('type', 'unknown')
            node_label = node_data.get('label', node_id)
            display_lines.append(f"{node_id} | {node_label} | {node_type}")
            
            # Show observations
            observations = node_data.get('observations', [])
            if observations:
                obs_strs = []
                for obs in observations[:5]:  # First 5
                    if isinstance(obs, dict):
                        desc = obs.get('description', obs.get('content', str(obs)))
                        obs_strs.append(desc)
                    else:
                        obs_strs.append(str(obs))
                if obs_strs:
                    display_lines.append(f"  obs: {'; '.join(obs_strs)}")
            
            # Show assumptions
            assumptions = node_data.get('assumptions', [])
            if assumptions:
                assum_strs = []
                for assum in assumptions[:3]:  # First 3
                    if isinstance(assum, dict):
                        desc = assum.get('description', assum.get('content', str(assum)))
                        assum_strs.append(desc)
                    else:
                        assum_strs.append(str(assum))
                if assum_strs:
                    display_lines.append(f"  assume: {'; '.join(assum_strs)}")
            
            # Show FULL code if present - agent needs to see everything for analysis
            if 'cards' in node_data and node_data['cards']:
                display_lines.append(f"  === CODE ({len(node_data['cards'])} blocks) ===")
                for i, card in enumerate(node_data['cards']):
                    content = card.get('content', '')
                    if content:
                        card_type = card.get('type', 'code')
                        metadata = card.get('metadata', {})
                        relpath = metadata.get('relpath', 'unknown')
                        line_start = metadata.get('line_start', '?')
                        line_end = metadata.get('line_end', '?')
                        
                        display_lines.append(f"  --- Block {i+1} ({card_type}) from {relpath}:{line_start}-{line_end} ---")
                        # Show FULL content - no truncation
                        for line in content.split('\n'):
                            display_lines.append(f"    {line}")
                        display_lines.append("")  # Empty line between code blocks
            
            display_lines.append("")  # Empty line between nodes
        
        if not_found:
            display_lines.append(f"\n⚠️ ERROR - These nodes do not exist in {graph_name}: {', '.join(not_found)}")
            display_lines.append(f"Available nodes in {graph_name}:")
            # Show first 10 available nodes as examples
            available_nodes = list(node_by_id.keys())[:10]
            for node_id in available_nodes:
                display_lines.append(f"  • {node_id}")
            if len(node_by_id) > 10:
                display_lines.append(f"  ... and {len(node_by_id) - 10} more")
            display_lines.append("\nUse EXACT node IDs as shown above. Do not guess or modify node names!")
        
        nodes_display = '\n'.join(display_lines)
        
        # Aggregate card IDs across loaded nodes
        all_card_ids = []
        for nid in current_request_nodes:
            node_data = self.loaded_data['nodes'].get(nid, {})
            for c in node_data.get('cards', []) or []:
                cid = c.get('card_id') or (c.get('metadata') or {}).get('id')
                if cid:
                    all_card_ids.append(str(cid))
        # Deduplicate
        all_card_ids = list({cid for cid in all_card_ids if cid})

        return {
            'status': 'success',
            'summary': f'Loaded {current_loaded_count} nodes ({current_code_count} with code)',
            'nodes_display': nodes_display,
            'loaded_node_ids': current_request_nodes,
            'card_ids': all_card_ids
        }
    
    
    def _save_graph_updates(self, graph_name: str, graph_data: Dict):
        """Save graph updates back to disk using concurrent-safe GraphStore."""
        try:
            if graph_name in self.available_graphs:
                graph_path = Path(self.available_graphs[graph_name]['path'])
                
                # Use GraphStore for atomic save with built-in locking
                graph_store = GraphStore(graph_path, agent_id=self.agent_id)
                return graph_store.save_graph(graph_data)
                        
        except Exception as e:
            print(f"[!] Failed to save graph {graph_name}: {e}")
            return False
    
    def _reload_graph(self, graph_name: str):
        """Reload a graph from disk using concurrent-safe GraphStore."""
        try:
            if graph_name in self.available_graphs:
                graph_path = Path(self.available_graphs[graph_name]['path'])
                
                # Use GraphStore for atomic read with built-in locking
                graph_store = GraphStore(graph_path, agent_id=self.agent_id)
                return graph_store.load_graph()
                    
        except Exception as e:
            print(f"[!] Failed to reload graph {graph_name}: {e}")
            return None
    
    def _update_node(self, params: Dict) -> Dict:
        """Update a node with observations or assumptions about its behavior."""
        node_id = params.get('node_id')
        if not node_id:
            # Check if user mistakenly passed node_ids (plural)
            if params.get('node_ids'):
                return {'status': 'error', 'error': 'update_node requires node_id (singular), not node_ids. Update one node at a time.'}
            return {'status': 'error', 'error': 'node_id is required for update_node action'}
        
        # First refresh all loaded graphs to get latest updates from other agents
        self._refresh_loaded_graphs()
        
        # Check if node exists in loaded data or graphs
        found = False
        if node_id in self.loaded_data.get('nodes', {}):
            found = True
        else:
            # Try to find it in graphs (nodes are stored as a list)
            for graph_name, graph_data in self._iterate_graphs():
                nodes = graph_data.get('nodes', [])
                for node in nodes:
                    if node.get('id') == node_id:
                        found = True
                        break
                if found:
                    break
            
        if not found:
            return {'status': 'error', 'error': f'Node {node_id} not found in any loaded graph'}
        
        # Update the node in the graph(s)
        updated_graphs = []
        observations = params.get('observations') or []
        assumptions = params.get('assumptions') or []
        
        for graph_name, graph_data in self._iterate_graphs():
            nodes = graph_data.get('nodes', [])
            for node in nodes:
                if node.get('id') != node_id:
                    continue
                
                # Initialize fields if not present
                if 'observations' not in node:
                    node['observations'] = []
                if 'assumptions' not in node:
                    node['assumptions'] = []
                
                # Add new observations (simplified - strings only as per prompt)
                for obs in observations:
                    if isinstance(obs, str):
                        node['observations'].append(obs)
                    elif isinstance(obs, dict):
                        # If dict provided, extract description
                        node['observations'].append(obs.get('description', str(obs)))
                
                # Add new assumptions (simplified - strings only as per prompt)
                for assum in assumptions:
                    if isinstance(assum, str):
                        node['assumptions'].append(assum)
                    elif isinstance(assum, dict):
                        # If dict provided, extract description
                        node['assumptions'].append(assum.get('description', str(assum)))
                
                # Save the updated graph to disk for sharing
                self._save_graph_updates(graph_name, graph_data)
                
                updated_graphs.append(graph_name)
                break  # Found and updated the node
        
        if not updated_graphs:
            return {'status': 'error', 'error': f'Node {node_id} not found in any loaded graph'}
        
        obs_count = len(observations)
        assum_count = len(assumptions)
        
        return {
            'status': 'success',
            'summary': f"Updated node {node_id}: {obs_count} observations, {assum_count} assumptions",
            'graphs_updated': updated_graphs
        }
    
    def _deep_think(self) -> Dict:
        """Delegate deep analysis to the Strategist and form hypotheses accordingly."""
        try:
            context = self._build_context()
            from .strategist import Strategist
            strategist = Strategist(config=self.config or {})
            items = strategist.deep_think(context=context) or []
            added = 0
            guidance_model_info = None
            if hasattr(self, 'guidance_client') and self.guidance_client:
                try:
                    guidance_model_info = f"{self.guidance_client.provider_name}:{self.guidance_client.model}"
                except Exception:
                    guidance_model_info = None
            for it in items:
                params = {
                    'description': it.get('description', 'Hypothesis'),
                    'details': it.get('details', ''),
                    'vulnerability_type': it.get('vulnerability_type', 'security_issue'),
                    'severity': it.get('severity', 'medium'),
                    'confidence': it.get('confidence', 0.6),
                    'node_ids': it.get('node_ids', ['system']),
                    'reasoning': it.get('reasoning', ''),
                    'graph_name': 'SystemArchitecture',
                    'guidance_model': guidance_model_info,
                }
                res = self._form_hypothesis(params)
                if res.get('status') == 'success':
                    added += 1
            return {'status': 'success', 'summary': f'Deep analysis added {added} hypotheses', 'hypotheses_formed': added}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    
    def _guess_vuln_type(self, text: str) -> str:
        """Guess vulnerability type from description."""
        text_lower = text.lower()
        if 'reentran' in text_lower:
            return 'reentrancy'
        elif 'overflow' in text_lower or 'underflow' in text_lower:
            return 'integer_overflow'
        elif 'auth' in text_lower or 'access' in text_lower or 'permission' in text_lower:
            return 'access_control'
        elif 'dos' in text_lower or 'denial' in text_lower:
            return 'denial_of_service'
        elif 'inject' in text_lower:
            return 'injection'
        else:
            return 'security_issue'
    
    def _form_hypothesis(self, params: Dict) -> Dict:
        """Form a new hypothesis."""
        from .concurrent_knowledge import Hypothesis
        
        # Ensure we have at least one node ID
        node_ids = params.get('node_ids') or []
        if not node_ids:
            return {'status': 'error', 'error': 'Hypothesis must reference at least one node'}
        
        # Determine which graph this hypothesis relates to
        graph_name = params.get('graph_name', 
                               self.loaded_data.get('system_graph', {}).get('name', 'unknown'))
        
        # Get model information
        model_info = f"{self.llm.provider_name}:{self.llm.model}"
        
        # Check if this is from deep_think (params will have guidance_model set)
        junior_model = model_info  # Default to agent model
        senior_model = params.get('guidance_model')  # Set if from deep_think
        
        # Create hypothesis object with compact title but detailed description
        # The title is compact but the description must be COMPLETE
        hypothesis = Hypothesis(
            title=params.get('description', 'vuln')[:60],  # Keep title compact for display
            description=params.get('details', params.get('description', '')),  # FULL details here
            vulnerability_type=params.get('vulnerability_type', 'unknown'),
            severity=params.get('severity', 'medium'),
            confidence=params.get('confidence', 0.5),
            node_refs=node_ids,
            reasoning=params.get('reasoning', ''),
            created_by=self.agent_id,
            reported_by_model=senior_model or junior_model,  # Legacy field for backward compatibility
            junior_model=junior_model,
            senior_model=senior_model,
            session_id=getattr(self, 'session_id', None),
            visibility=params.get('visibility', getattr(self, 'default_hypothesis_visibility', 'global'))
        )
        
        # Extract source files from nodes
        source_files = set()
        affected_functions = []
        
        # Look up source files for each node
        for node_id in node_ids:
            # Check in system graph
            if self.loaded_data.get('system_graph'):
                for node in self.loaded_data['system_graph']['data'].get('nodes', []):
                    if node.get('id') == node_id:
                        # Extract source files from source_refs (card IDs)
                        for card_id in node.get('source_refs', []):
                            if card_id in self.cards:
                                card = self.cards[card_id]
                                # Cards have 'relpath' not 'file_path'
                                if 'relpath' in card:
                                    source_files.add(card['relpath'])
                        
                        # Track function name if it's a function node
                        if node.get('type') == 'function':
                            func_name = node.get('label', node_id).split('.')[-1]
                            affected_functions.append(func_name)
            
            # Also check in loaded graphs
            for graph_data in self.loaded_data.get('graphs', {}).values():
                for node in graph_data.get('nodes', []):
                    if node.get('id') == node_id:
                        for card_id in node.get('source_refs', []):
                            if card_id in self.cards:
                                card = self.cards[card_id]
                                # Cards have 'relpath' not 'file_path'
                                if 'relpath' in card:
                                    source_files.add(card['relpath'])
        
        # Store graph name and source files in properties (NOT shown to agent)
        hypothesis.properties = {
            'graph_name': graph_name,
            'source_files': list(source_files),
            'affected_functions': affected_functions
        }
        
        # Store in persistent hypothesis store
        success, hyp_id = self.hypothesis_store.propose(hypothesis)
        
        # Also keep in memory for backward compatibility
        self.loaded_data['hypotheses'].append({
            'id': hyp_id,
            'description': hypothesis.title,
            'vulnerability_type': hypothesis.vulnerability_type,
            'confidence': hypothesis.confidence,
            'status': hypothesis.status,
            'node_ids': hypothesis.node_refs,
            'evidence': []
        })
        
        return {
            'status': 'success' if success else 'error',
            'summary': f"Formed hypothesis: {hypothesis.title}" if success else f"Failed: {hyp_id}",
            'hypothesis_id': hyp_id if success else None,
            'hypothesis_index': len(self.loaded_data['hypotheses']) - 1
        }
    
    def _update_hypothesis(self, params: Dict) -> Dict:
        """Update an existing hypothesis."""
        from .concurrent_knowledge import Evidence
        
        # Support both index and ID
        hyp_id = params.get('hypothesis_id')
        if not hyp_id:
            index = params.get('hypothesis_index', 0)
            if index >= len(self.loaded_data['hypotheses']):
                return {'status': 'error', 'error': 'Invalid hypothesis index'}
            hyp_id = self.loaded_data['hypotheses'][index].get('id')
        
        # Update confidence if provided
        if 'new_confidence' in params and hyp_id:
            reason = params.get('reason', 'Agent analysis')
            self.hypothesis_store.adjust_confidence(hyp_id, params['new_confidence'], reason)
            
            # Update in memory too
            for h in self.loaded_data['hypotheses']:
                if h.get('id') == hyp_id:
                    h['confidence'] = params['new_confidence']
        
        # Add evidence if provided
        if 'evidence' in params and hyp_id:
            evidence = Evidence(
                description=params['evidence'],
                type=params.get('evidence_type', 'supports'),
                confidence=params.get('evidence_confidence', 0.7),
                node_refs=params.get('node_ids') or [],
                created_by=self.agent_id
            )
            self.hypothesis_store.add_evidence(hyp_id, evidence)
            
            # Update in memory
            for h in self.loaded_data['hypotheses']:
                if h.get('id') == hyp_id:
                    h['evidence'].append(params['evidence'])
        
        return {
            'status': 'success',
            'summary': f"Updated hypothesis {hyp_id}",
            'hypothesis_id': hyp_id
        }
    
    def _generate_report(self, iterations: int) -> Dict:
        """Generate final investigation report."""
        # Categorize hypotheses
        confirmed = [h for h in self.loaded_data['hypotheses'] if h['confidence'] >= 0.8]
        rejected = [h for h in self.loaded_data['hypotheses'] if h['confidence'] <= 0.2]
        uncertain = [h for h in self.loaded_data['hypotheses'] 
                    if 0.2 < h['confidence'] < 0.8]
        
        return {
            'investigation_goal': self.investigation_goal,
            'iterations_completed': iterations,
            'graphs_analyzed': list(self.loaded_data.get('graphs', {}).keys()),
            'nodes_analyzed': len(self.loaded_data['nodes']),
            'hypotheses': {
                'total': len(self.loaded_data['hypotheses']),
                'confirmed': len(confirmed),
                'rejected': len(rejected),
                'uncertain': len(uncertain)
            },
            'detailed_hypotheses': [
                {
                    'description': h['description'],
                    'type': h['vulnerability_type'],
                    'confidence': h['confidence'],
                    'status': 'confirmed' if h['confidence'] >= 0.8 
                             else 'rejected' if h['confidence'] <= 0.2 
                             else 'uncertain',
                    'evidence': h.get('evidence', [])
                }
                for h in self.loaded_data['hypotheses']
            ]
        }
