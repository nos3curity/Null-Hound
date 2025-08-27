"""
Agent command for autonomous security analysis.
"""

import json
import click
import time
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.agent import AutonomousAgent
from analysis.run_tracker import RunTracker
from llm.token_tracker import get_token_tracker
from pydantic import BaseModel, Field

def get_project_dir(project_id: str) -> Path:
    """Get project directory path."""
    return Path.home() / ".hound" / "projects" / project_id

def run_investigation(project_path: str, prompt: str, iterations: Optional[int] = None, config_path: Optional[Path] = None, debug: bool = False, platform: Optional[str] = None, model: Optional[str] = None):
    """Run a user-driven investigation."""
    console = Console()
    
    # Load config properly
    from commands.graph import load_config
    config = None
    
    try:
        if config_path:
            # Use provided config path
            if config_path.exists():
                config = load_config(config_path)
        else:
            # Try default config.yaml
            default_config = Path("config.yaml")
            if default_config.exists():
                config = load_config(default_config)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
        console.print("[yellow]Using default configuration[/yellow]")
        config = None
    
    # If no config was loaded but platform/model were provided, create minimal config
    if not config and (platform or model):
        config = {'models': {'agent': {}}}
    
    # Override platform and model if provided
    if config and (platform or model):
        # Ensure the models.agent structure exists
        if 'models' not in config:
            config['models'] = {}
        if 'agent' not in config['models']:
            config['models']['agent'] = {}
        
        if platform:
            config['models']['agent']['provider'] = platform
            console.print(f"[cyan]Overriding agent provider: {platform}[/cyan]")
        if model:
            config['models']['agent']['model'] = model
            console.print(f"[cyan]Overriding agent model: {model}[/cyan]")
    
    # Resolve project path
    if '/' in project_path or Path(project_path).exists():
        project_dir = Path(project_path).resolve()
    else:
        project_dir = get_project_dir(project_path)
    
    # Look for graphs and manifest
    graphs_dir = project_dir / "graphs"
    manifest_dir = project_dir / "manifest"
    
    # Check for knowledge_graphs.json or individual graph files
    knowledge_graphs_path = graphs_dir / "knowledge_graphs.json"
    if not knowledge_graphs_path.exists():
        # Create it from available graphs
        graph_files = list(graphs_dir.glob("graph_*.json"))
        if not graph_files:
            console.print("[red]Error: No graphs found. Run 'graph build' first.[/red]")
            return
        
        # Create knowledge_graphs.json
        graphs_dict = {}
        for graph_file in graph_files:
            graph_name = graph_file.stem.replace('graph_', '')
            graphs_dict[graph_name] = str(graph_file)
        
        with open(knowledge_graphs_path, 'w') as f:
            json.dump({'graphs': graphs_dict}, f, indent=2)
    
    # Initialize agent
    console.print("[cyan]Initializing agent...[/cyan]")
    agent = AutonomousAgent(
        graphs_metadata_path=knowledge_graphs_path,
        manifest_path=manifest_dir,
        agent_id=f"investigate_{int(time.time())}",
        config=config,  # Pass the loaded config dict, not the path
        debug=debug
    )
    
    # Run investigation with live display
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    
    # Create a live display with rolling event log
    from rich.markdown import Markdown
    from datetime import datetime
    
    event_log = []  # list of strings (renderables)
    
    def _shorten(s: str, n: int = 140) -> str:
        return (s[: n - 3] + '...') if isinstance(s, str) and len(s) > n else (s or '')
    
    def _format_params(p) -> str:
        try:
            return _shorten(json.dumps(p, separators=(',', ':'), ensure_ascii=False), 160)
        except Exception:
            return "{}"
    
    def _panel_from_events():
        # keep last 8 events
        lines = event_log[-8:] if len(event_log) > 8 else event_log
        content = "\n".join(lines) if lines else "Initializing investigation..."
        return Panel(content, title="[bold cyan]Investigation Progress[/bold cyan]", border_style="cyan")
    
    def update_progress(info):
        """Update the live display with current status and reasoning."""
        status = info.get('status', '')
        message = info.get('message', '')
        iteration = info.get('iteration', 0)
        now = datetime.now().strftime('%H:%M:%S')
        
        if status == 'analyzing':
            event_log.append(f"[yellow]{now}[/yellow] [bold]Iter {iteration}[/bold] [yellow]Analyzing[/yellow]: {message}")
        elif status == 'decision':
            action = info.get('action', '-')
            reasoning = info.get('reasoning', '')  # Don't abbreviate thoughts
            params = _format_params(info.get('parameters', {}))
            event_log.append(
                f"[cyan]{now}[/cyan] [bold]Iter {iteration}[/bold] [cyan]Decision[/cyan]: action={action}\n"
                f"  [dim]Thought:[/dim] {reasoning}\n  [dim]Params:[/dim] {params}"
            )
        elif status == 'executing':
            event_log.append(f"[blue]{now}[/blue] [bold]Iter {iteration}[/bold] [blue]Executing[/blue]: {message}")
        elif status == 'result':
            res = info.get('result', {}) or {}
            summary = res.get('summary') or res.get('status') or message
            event_log.append(f"[green]{now}[/green] [bold]Iter {iteration}[/bold] [green]Result[/green]: {_shorten(summary, 160)}")
        elif status == 'hypothesis_formed':
            event_log.append(f"[green]{now}[/green] [bold]Iter {iteration}[/bold] [green]Hypothesis[/green]: {message}")
        elif status == 'code_loaded':
            event_log.append(f"[blue]{now}[/blue] [bold]Iter {iteration}[/bold] [blue]Code Loaded[/blue]: {message}")
        elif status == 'generating_report':
            event_log.append(f"[magenta]{now}[/magenta] [bold]Iter {iteration}[/bold] [magenta]Report[/magenta]: {message}")
        elif status == 'complete':
            event_log.append(f"[bold green]{now}[/bold green] [bold]Iter {iteration}[/bold] [bold green]Complete[/bold green]: {message}")
        else:
            event_log.append(f"[white]{now}[/white] [bold]Iter {iteration}[/bold] [white]{status or 'Working'}[/white]: {message}")
        
        live.update(_panel_from_events())
    
    with Live(_panel_from_events(), console=console, refresh_per_second=6, transient=True) as live:
        try:
            # Execute investigation with progress callback
            max_iters = iterations if iterations and iterations > 0 else 10
            report = agent.investigate(prompt, max_iterations=max_iters, progress_callback=update_progress)
            
            # Clear the live display
            live.stop()
            
            # Display results
            display_investigation_report(report)
            
            # Finalize debug log if in debug mode
            if debug and agent.debug_logger:
                log_path = agent.debug_logger.finalize(summary={
                    'total_iterations': report.get('iterations_completed', 0),
                    'hypotheses_tested': report['hypotheses']['total'],
                    'confirmed': report['hypotheses']['confirmed'],
                    'rejected': report['hypotheses']['rejected']
                })
                console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")
                console.print(f"[dim]Open in browser: file://{log_path}[/dim]")
                try:
                    import webbrowser
                    webbrowser.open(f"file://{log_path}")
                except Exception:
                    pass
            
        except Exception as e:
            console.print(f"[red]Investigation failed: {e}[/red]")
            if debug:
                import traceback
                console.print(traceback.format_exc())
                if agent and agent.debug_logger:
                    log_path = agent.debug_logger.finalize()
                    console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")
                    console.print(f"[dim]Open in browser: file://{log_path}[/dim]")
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{log_path}")
                    except Exception:
                        pass

def display_investigation_report(report: dict):
    """Display investigation report in a nice format."""
    console = Console()
    
    # Header
    console.print("\n[bold magenta]═══ INVESTIGATION REPORT ═══[/bold magenta]\n")
    
    # Goal and summary
    console.print(f"[bold]Investigation Goal:[/bold] {report['investigation_goal']}")
    console.print(f"[bold]Iterations:[/bold] {report['iterations_completed']}")
    console.print()
    
    # Hypothesis summary
    hyp_stats = report['hypotheses']
    console.print(f"[bold cyan]Hypotheses:[/bold cyan]")
    console.print(f"  • Total: {hyp_stats['total']}")
    console.print(f"  • [green]Confirmed: {hyp_stats['confirmed']}[/green]")
    console.print(f"  • [red]Rejected: {hyp_stats['rejected']}[/red]")
    console.print(f"  • [yellow]Uncertain: {hyp_stats['uncertain']}[/yellow]")
    console.print()
    
    # Detailed hypotheses
    if report.get('detailed_hypotheses'):
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Hypothesis", style="white", width=50, overflow="fold")
        table.add_column("Model", style="dim", width=20)
        table.add_column("Confidence", justify="center")
        table.add_column("Status", justify="center")
        
        for hyp in report['detailed_hypotheses']:
            # Color-code confidence
            conf = hyp['confidence']
            if conf >= 0.8:
                conf_style = "[bold green]"
            elif conf <= 0.2:
                conf_style = "[bold red]"
            else:
                conf_style = "[yellow]"
            
            # Status styling
            status = hyp['status']
            if status == 'confirmed':
                status_style = "[bold green]CONFIRMED[/bold green]"
            elif status == 'rejected':
                status_style = "[bold red]REJECTED[/bold red]"
            else:
                status_style = "[yellow]TESTING[/yellow]"
            
            # Get model info, fallback to "unknown" if not present
            model = hyp.get('reported_by_model', 'unknown')
            
            # Use full description - Rich will handle wrapping with overflow="fold"
            table.add_row(
                hyp['description'],  # Show full description
                model,
                f"{conf_style}{conf*100:.0f}%[/{conf_style.strip('[')}",
                status_style
            )
        
        console.print(table)
        console.print()
    
    # Conclusion
    console.print(f"[bold]Conclusion:[/bold]")
    conclusion = report.get('conclusion', 'No conclusion available')
    if 'LIKELY TRUE' in conclusion:
        console.print(f"  [green]{conclusion}[/green]")
    elif 'LIKELY FALSE' in conclusion:
        console.print(f"  [red]{conclusion}[/red]")
    else:
        console.print(f"  [yellow]{conclusion}[/yellow]")
    console.print()
    
    # Summary narrative
    if report.get('summary'):
        console.print("[bold]Summary:[/bold]")
        console.print(Panel(report['summary'], border_style="dim"))
    
    console.print("\n[dim]Investigation complete.[/dim]")


console = Console()


def format_tool_call(call):
    """Format a tool call for pretty display."""
    params_str = json.dumps(call.parameters, indent=2) if call.parameters else "{}"
    
    # Use different colors for different tool types
    tool_colors = {
        'focus': 'cyan',
        'query_graph': 'blue',
        'update_node': 'yellow',
        'propose_hypothesis': 'red',
        'update_hypothesis': 'magenta',
        'add_edge': 'green',
        'summarize': 'white'
    }
    
    color = tool_colors.get(call.tool_name, 'white')
    
    content = f"[bold]{call.description}[/bold]\n\n"
    if hasattr(call, 'reasoning') and call.reasoning:
        content += f"[italic yellow]Reasoning: {call.reasoning}[/italic yellow]\n\n"
    content += f"[dim]Tool:[/dim] [{color}]{call.tool_name}[/{color}]\n"
    if hasattr(call, 'priority'):
        content += f"[dim]Priority:[/dim] {call.priority}/10\n"
    content += f"[dim]Parameters:[/dim]\n{params_str}"
    
    return Panel(
        content,
        title=f"[bold cyan]Tool Call[/bold cyan]",
        border_style="cyan"
    )


def format_tool_result(result):
    """Format tool execution result."""
    if result.get('status') == 'success':
        style = "green"
        icon = "✓"
    else:
        style = "red"
        icon = "✗"
    
    # Extract key information based on result content
    details = []
    if 'focused_nodes' in result:
        details.append(f"Focused on {result['focused_nodes']} nodes")
    if 'code_cards_loaded' in result:
        details.append(f"Loaded {result['code_cards_loaded']} code cards")
    if 'matches' in result:
        details.append(f"Found {len(result['matches'])} matches")
    if 'hypothesis_id' in result:
        details.append(f"Hypothesis: {result['hypothesis_id'][:8]}...")
    if 'updates' in result:
        details.append(f"Applied {len(result['updates'])} updates")
    
    details_str = "\n".join(f"  • {d}" for d in details) if details else json.dumps(result, indent=2)
    
    return Panel(
        f"[{style}]{icon} {result.get('status', 'unknown').upper()}[/{style}]\n\n{details_str}",
        title="[bold]Result[/bold]",
        border_style=style
    )


def display_planning_phase(agent, items):
    """Display the planning phase output for investigations or tool calls."""
    
    # Check if we have investigations or tool calls
    if items and hasattr(items[0], 'goal'):  # Investigation objects
        console.print("\n[bold cyan]═══ INVESTIGATION PLANNING ═══[/bold cyan]")
        console.print(f"Planning [bold]{len(items)}[/bold] investigations...\n")
        
        # Create a table of planned investigations
        table = Table(title="High-Level Investigations", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Goal", style="cyan", width=50)
        table.add_column("Focus Areas", style="yellow", width=30)
        table.add_column("Priority", justify="center")
        
        for i, inv in enumerate(items, 1):
            # Color code priority
            if inv.priority >= 8:
                priority_style = "[bold red]"
            elif inv.priority >= 5:
                priority_style = "[yellow]"
            else:
                priority_style = "[dim]"
            
            table.add_row(
                str(i),
                inv.goal[:50],
                ', '.join(inv.focus_areas[:2]) if inv.focus_areas else "-",
                f"{priority_style}{inv.priority}[/]"
            )
        
        console.print(table)
        
        # Show reasoning for each investigation
        for i, inv in enumerate(items, 1):
            console.print(f"\n[bold]{i}. {inv.goal}[/bold]")
            console.print(f"   [dim]Reasoning: {inv.reasoning}[/dim]")
    
    else:  # ToolCall objects (backward compatibility)
        console.print("\n[bold cyan]═══ PLANNING PHASE ═══[/bold cyan]")
        console.print(f"Planning [bold]{len(items)}[/bold] next steps...\n")
        
        # Create a table of planned actions
        table = Table(title="Planned Actions", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Tool", style="cyan")
        table.add_column("Description", style="white", width=50)
        table.add_column("Reasoning", style="yellow", width=30)
        
        for i, call in enumerate(items, 1):
            table.add_row(
                str(i),
                call.tool_name,
                call.description[:50] if call.description else "-",
                call.reasoning[:30] if call.reasoning else "-"
            )
        
        console.print(table)


def display_execution_phase(call, result):
    """Display execution of a single tool call."""
    console.print("\n[bold green]═══ EXECUTING ═══[/bold green]")
    console.print(format_tool_call(call))
    console.print(format_tool_result(result))


def display_agent_summary(summary, time_limit_reached=False):
    """Display final agent summary with detailed findings."""
    
    # Header based on completion reason
    if time_limit_reached:
        console.print("\n[bold yellow]═══ TIME LIMIT REACHED - AGENT REPORT ═══[/bold yellow]\n")
    else:
        console.print("\n[bold magenta]═══ AGENT SUMMARY ═══[/bold magenta]\n")
    
    # Basic statistics
    summary_text = f"""
[bold]Agent ID:[/bold] {summary['agent_id']}
[bold]Iterations Completed:[/bold] {summary['iterations']}
[bold]Tool Calls Executed:[/bold] {summary['tool_calls_completed']}

[bold cyan]Graph Statistics:[/bold cyan]
  • Nodes Analyzed: {summary['graph_stats'].get('num_nodes', 0)}
  • Edges Traced: {summary['graph_stats'].get('num_edges', 0)}
  • Observations Added: {summary['graph_stats'].get('observations', 0)}
  • Invariants Added: {summary['graph_stats'].get('invariants', 0)}
  
[bold yellow]Security Findings:[/bold yellow]
  • Hypotheses Proposed: {summary['hypotheses']['total']}
  • Confirmed Vulnerabilities: {summary['hypotheses']['confirmed']}
"""
    
    console.print(Panel(summary_text, title="[bold]Statistics[/bold]", border_style="cyan"))
    
    # Detailed hypotheses if any exist
    if summary.get('all_hypotheses'):
        console.print("\n[bold red]VULNERABILITY HYPOTHESES:[/bold red]")
        
        # Create a table for hypotheses
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Node", style="yellow", width=20, overflow="ellipsis")
        table.add_column("Type", style="cyan", width=15)
        table.add_column("Description", style="white", width=45, overflow="fold")
        table.add_column("Model", style="dim", width=15, overflow="ellipsis")
        table.add_column("Confidence", justify="center")
        table.add_column("Status", justify="center")
        
        for hyp in summary['all_hypotheses'][:10]:  # Show top 10
            # Color code confidence
            conf = hyp.get('confidence', 0)
            if conf >= 0.8:
                conf_style = "[bold red]"
            elif conf >= 0.5:
                conf_style = "[yellow]"
            else:
                conf_style = "[dim]"
            
            # Status indicator
            status = hyp.get('status', 'investigating')
            if status == 'confirmed':
                status_display = "[bold red]CONFIRMED[/bold red]"
            elif status == 'rejected':
                status_display = "[dim]rejected[/dim]"
            else:
                status_display = "[yellow]investigating[/yellow]"
            
            # Get model info
            model = hyp.get('reported_by_model', 'unknown')
            
            # Show full description and let Rich handle wrapping
            table.add_row(
                hyp.get('id', 'unknown')[:12],
                hyp.get('node_id', 'unknown'),  # Will be ellipsized by Rich if too long
                hyp.get('vulnerability_type', 'unknown'),
                hyp.get('description', ''),  # Show full description
                model if model else 'unknown',  # Will be ellipsized by Rich if too long
                f"{conf_style}{conf:.2f}[/]",
                status_display
            )
        
        console.print(table)
    
    # Tool execution summary
    if summary.get('tool_execution_summary'):
        console.print("\n[bold green]TOOL EXECUTION SUMMARY:[/bold green]")
        
        tool_table = Table(show_header=True, header_style="bold cyan")
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Calls", justify="center")
        tool_table.add_column("Successful", justify="center", style="green")
        tool_table.add_column("Failed", justify="center", style="red")
        
        for tool_name, stats in summary['tool_execution_summary'].items():
            tool_table.add_row(
                tool_name,
                str(stats.get('total', 0)),
                str(stats.get('successful', 0)),
                str(stats.get('failed', 0))
            )
        
        console.print(tool_table)
    
    # Areas analyzed
    if summary.get('analyzed_areas'):
        console.print("\n[bold blue]AREAS ANALYZED:[/bold blue]")
        for area in summary['analyzed_areas']:
            console.print(f"  • {area['name']}: {area['description']}")
    
    # Key findings narrative
    if summary.get('key_findings'):
        console.print("\n[bold red]KEY FINDINGS:[/bold red]")
        for i, finding in enumerate(summary['key_findings'][:5], 1):
            console.print(f"\n  {i}. [yellow]{finding.get('title', 'Finding')}[/yellow]")
            console.print(f"     {finding.get('description', '')}")
            if finding.get('recommendation'):
                console.print(f"     [dim]Recommendation: {finding['recommendation']}[/dim]")
    
    console.print(Panel("", title="[bold]End of Report[/bold]", border_style="magenta"))


class AgentRunner:
    """Manages agent execution with beautiful output."""
    
    def __init__(self, project_id: str, config_path: Optional[Path] = None, 
                 iterations: Optional[int] = None, time_limit_minutes: Optional[int] = None,
                 debug: bool = False, platform: Optional[str] = None, model: Optional[str] = None):
        self.project_id = project_id
        self.config_path = config_path
        self.max_iterations = iterations
        self.time_limit_minutes = time_limit_minutes
        self.debug = debug
        self.platform = platform
        self.model = model
        self.agent = None
        self.start_time = None
        self.completed_investigations = []  # Track completed investigation goals
        self.run_tracker: Optional[RunTracker] = None
        
    def initialize(self):
        """Initialize the agent."""
        # First check if project_id is actually a path to a project directory
        if '/' in self.project_id or Path(self.project_id).exists():
            # It's a path to the project output
            project_dir = Path(self.project_id)
            if not project_dir.is_absolute():
                project_dir = project_dir.resolve()
        else:
            # It's a project ID, use default location
            project_dir = get_project_dir(self.project_id)
        
        # Look for the knowledge graphs metadata file
        graphs_dir = project_dir / "graphs"
        knowledge_graphs_path = graphs_dir / "knowledge_graphs.json"
        manifest_path = project_dir / "manifest"
        
        # If knowledge_graphs.json doesn't exist, look for any graph file
        if knowledge_graphs_path.exists():
            # Use the SystemOverview graph or first available graph
            with open(knowledge_graphs_path, 'r') as f:
                graphs_meta = json.load(f)
            if graphs_meta.get('graphs'):
                graphs_dict = graphs_meta['graphs']
                # Prefer SystemOverview if it exists
                if 'SystemOverview' in graphs_dict:
                    graph_path = Path(graphs_dict['SystemOverview'])
                else:
                    # Use the first available graph
                    graph_name = list(graphs_dict.keys())[0]
                    graph_path = Path(graphs_dict[graph_name])
                console.print(f"[green]Using graph: {graph_path.name}[/green]")
            else:
                console.print("[red]Error:[/red] No graphs found in knowledge_graphs.json")
                return False
        elif graphs_dir.exists():
            # Fallback: look for any graph_*.json file, preferably SystemOverview
            graph_files = list(graphs_dir.glob("graph_*.json"))
            if graph_files:
                # Prefer SystemOverview if it exists
                system_overview = graphs_dir / "graph_SystemOverview.json"
                if system_overview.exists():
                    graph_path = system_overview
                else:
                    graph_path = graph_files[0]
                console.print(f"[yellow]Using graph: {graph_path.name}[/yellow]")
            else:
                console.print(f"[red]Error:[/red] No graph files found in {graphs_dir}")
                return False
        else:
            console.print(f"[red]Error:[/red] No graphs directory found at {graphs_dir}")
            console.print("[yellow]Run 'hound build' first or check the path.[/yellow]")
            return False
        
        if not graph_path.exists():
            console.print(f"[red]Error:[/red] Graph file not found: {graph_path}")
            return False
        
        # Load config properly using the standard method
        from commands.graph import load_config
        if self.config_path and self.config_path.exists():
            config = load_config(self.config_path)
        else:
            config = load_config()  # Uses default config.yaml
        
        # Override platform and model if provided
        if self.platform or self.model:
            # Ensure the models.agent structure exists
            if 'models' not in config:
                config['models'] = {}
            if 'agent' not in config['models']:
                config['models']['agent'] = {}
            
            if self.platform:
                config['models']['agent']['provider'] = self.platform
                console.print(f"[cyan]Overriding agent provider: {self.platform}[/cyan]")
            if self.model:
                config['models']['agent']['model'] = self.model
                console.print(f"[cyan]Overriding agent model: {self.model}[/cyan]")
        
        # Keep config for planning
        self.config = config
        
        # Create agent with knowledge graphs metadata
        self.agent = AutonomousAgent(
            graphs_metadata_path=knowledge_graphs_path,
            manifest_path=manifest_path,
            agent_id=f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,  # Pass the loaded config dict
            debug=self.debug
        )
        
        # Set debug flag
        self.agent.debug = self.debug
        
        if self.max_iterations:
            self.agent.max_iterations = self.max_iterations
        
        return True
    
    def _graph_summary(self, max_nodes: int = 20) -> str:
        """Create a compact summary of the system graph for planning."""
        try:
            g = self.agent.loaded_data.get('system_graph', {}).get('data', {})
            nodes = g.get('nodes', [])
            edges = g.get('edges', [])
            parts = [
                f"Nodes: {len(nodes)}",
                f"Edges: {len(edges)}"
            ]
            # List first N node labels/types
            for n in nodes[:max_nodes]:
                nid = n.get('id', '')
                lbl = n.get('label') or nid
                typ = n.get('type', '')
                parts.append(f"- {lbl} ({typ})")
            return "\n".join(parts)
        except Exception:
            return "(no graph summary available)"

    def _plan_investigations(self, n: int) -> List[object]:
        """Use the LLM to propose the next n investigations.
        Preference: high-level aspects to review unless the graph itself suggests a specific risk.
        Filters out already completed investigations.
        """
        from llm.unified_client import UnifiedLLMClient

        class InvestigationItem(BaseModel):
            goal: str = Field(description="Investigation goal or question")
            focus_areas: List[str] = Field(default_factory=list)
            priority: int = Field(ge=1, le=10, description="1-10, 10 = highest")
            reasoning: str = Field(default="", description="Rationale for why this is promising")
            category: str = Field(default="aspect", description="aspect | suspicion")
            expected_impact: str = Field(default="medium", description="high | medium | low")

        class InvestigationPlan(BaseModel):
            investigations: List[InvestigationItem] = Field(
                default_factory=list,
                description=f"List of exactly {n} investigation items to plan"
            )

        llm = UnifiedLLMClient(cfg=self.config, profile="guidance")

        system = (
            "You are a senior smart-contract security auditor planning an audit roadmap.\n"
            "Plan the next investigations based on the system architecture graph.\n\n"
            "GUIDELINES:\n"
            "- Prefer HIGH-LEVEL aspects to review (e.g., authorization/roles, initialization/ownership, external call surfaces, token supply/minting, signature verification & nonces, oracles, withdrawals/escrow, pausing, upgrade paths).\n"
            "- Only propose a SPECIFIC vulnerability suspicion if the graph itself strongly suggests it (and explain the evidence).\n"
            "- Prioritize by expected impact and error-proneness.\n"
            "- Be concise and neutral in goals for aspects (e.g., 'Review authorization and roles'); avoid speculative claims unless evidence exists.\n\n"
            f"OUTPUT: Return EXACTLY {n} investigations, sorted by priority. Each item MUST include:\n"
            "  - goal (concise)\n  - focus_areas\n  - priority (1-10)\n  - expected_impact (high/medium/low)\n  - category ('aspect' or 'suspicion')\n  - reasoning (brief rationale; if 'suspicion', cite the graph evidence).\n"
            f"IMPORTANT: You MUST provide exactly {n} investigation items, no more, no less."
        )
        # Include completed investigations in context
        completed_str = ""
        if self.completed_investigations:
            completed_str = "\n\nAlready Completed Investigations:\n" + "\n".join(
                f"- {goal}" for goal in self.completed_investigations
            )
        
        user = (
            "System Graph Summary:\n" + self._graph_summary() +
            completed_str +
            f"\n\nPlan the top {n} NEW investigations (avoid repeating completed ones)."
        )
        def _heuristic_plan(k: int) -> List[object]:
            """Deterministic planner from system graph contents."""
            g = (self.agent.loaded_data.get('system_graph') or {}).get('data') or {}
            nodes = g.get('nodes', []) or []
            # Build a flat list of (label, type)
            labels = []
            for n in nodes:
                lbl = n.get('label') or n.get('id') or ''
                typ = (n.get('type') or '').lower()
                if lbl:
                    labels.append((str(lbl), typ))
            # Topics with keywords
            topics = [
                ("Authorization and roles", ["auth", "role", "owner", "only", "admin", "access", "permission", "authority"], 9, "high"),
                ("Initialization and ownership transfer", ["init", "initialize", "constructor", "instantiate", "owner", "transfer"], 8, "high"),
                ("Token mint/burn and supply", ["mint", "burn", "supply", "token"], 8, "high"),
                ("External calls and reentrancy", ["call", "transfer", "send", "reentr", "callback", "hook"], 7, "high"),
                ("Signature verification and nonces", ["signature", "sign", "verify", "ecdsa", "ed25519", "nonce"], 7, "high"),
                ("Oracles and pricing", ["oracle", "price", "feed"], 6, "medium"),
                ("Withdrawals/escrow and limits", ["withdraw", "escrow", "redeem", "claim"], 6, "high"),
                ("Pause/emergency stops", ["pause", "paused", "guardian"], 5, "medium"),
            ]
            scored = []
            import re
            for title, keys, base_prio, impact in topics:
                score = 0
                matches = []
                for lbl, typ in labels:
                    l = lbl.lower()
                    if any(k in l for k in keys):
                        score += 1
                        matches.append(lbl)
                if score > 0:
                    goal = f"Review {title.lower()} (focus on components: {', '.join(matches[:3])})"
                    item = type('Inv', (), {
                        'goal': goal,
                        'focus_areas': [title],
                        'priority': min(10, base_prio + min(score, 3)),
                        'reasoning': f"Graph indicates relevant components: {', '.join(matches[:5])}",
                        'category': 'aspect',
                        'expected_impact': impact,
                    })()
                    scored.append((item.priority, item))
            # Fallback generic if nothing matched
            if not scored:
                generic_titles = [
                    ("Authorization and roles", 9, "high"),
                    ("Initialization and ownership", 8, "high"),
                    ("Token mint/burn and supply", 8, "high"),
                    ("External calls and reentrancy", 7, "high"),
                    ("Signature verification and nonces", 7, "high"),
                ]
                for title, pr, impact in generic_titles[:k]:
                    goal = f"Review {title.lower()}"
                    scored.append((pr, type('Inv', (), {
                        'goal': goal,
                        'focus_areas': [title],
                        'priority': pr,
                        'reasoning': 'Default high-impact area',
                        'category': 'aspect',
                        'expected_impact': impact,
                    })()))
            # Sort by priority desc, take top k unique by goal
            scored.sort(key=lambda x: -x[0])
            uniq = []
            seen = set()
            for _, it in scored:
                if it.goal not in seen:
                    seen.add(it.goal)
                    uniq.append(it)
                if len(uniq) >= k:
                    break
            return uniq

        try:
            if self.debug:
                console.print(f"[dim]Calling LLM with system prompt length: {len(system)}[/dim]")
                console.print(f"[dim]User prompt length: {len(user)}[/dim]")
            
            plan = llm.parse(system=system, user=user, schema=InvestigationPlan)
            items = list(plan.investigations) if plan and hasattr(plan, 'investigations') else []
            
            console.print(f"[cyan]LLM returned {len(items)} investigations[/cyan]")
            if self.debug and items:
                for i, item in enumerate(items[:3]):
                    console.print(f"[dim]  {i+1}. {getattr(item, 'goal', 'No goal')}[/dim]")
            
            # THIS SHOULD NOT HAPPEN - LLM should return n investigations
            if len(items) < n:
                console.print(f"[red]ERROR: LLM only returned {len(items)} investigations, requested {n}![/red]")
                console.print(f"[yellow]This indicates a problem with the LLM call. Falling back to heuristics.[/yellow]")
                fill = _heuristic_plan(n - len(items))
                items.extend(fill)
            # Filter out already completed investigations
            filtered_items = []
            for item in items:
                goal = getattr(item, 'goal', '')
                # Check if this goal has already been investigated (fuzzy match)
                is_duplicate = False
                for completed in self.completed_investigations:
                    if goal.lower() in completed.lower() or completed.lower() in goal.lower():
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_items.append(item)
            
            # Sort by priority desc, then by goal
            filtered_items.sort(key=lambda x: (-(getattr(x, 'priority', 0) or 0), getattr(x, 'goal', '')))
            
            # If we filtered too many, get more from heuristics
            if len(filtered_items) < n:
                extra = _heuristic_plan(n * 2)  # Get extra to filter from
                # Get existing goals to avoid duplicates
                existing_goals = {getattr(item, 'goal', '').lower() for item in filtered_items}
                
                for item in extra:
                    goal = getattr(item, 'goal', '')
                    # Check if already in filtered items
                    if goal.lower() in existing_goals:
                        continue
                    # Check if already completed
                    is_duplicate = False
                    for completed in self.completed_investigations:
                        if goal.lower() in completed.lower() or completed.lower() in goal.lower():
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        filtered_items.append(item)
                        existing_goals.add(goal.lower())
                        if len(filtered_items) >= n:
                            break
            
            return filtered_items[:n]
        except Exception as e:
            console.print(f"[red]Error in LLM planning: {str(e)}[/red]")
            console.print("[yellow]Falling back to heuristic planning[/yellow]")
            if self.debug:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            # Robust fallback: always return n items
            return _heuristic_plan(n)

    def _render_checklist(self, items: List[object], completed_index: int = -1):
        """Render a simple checklist; items up to completed_index are checked."""
        console.print("\n[bold cyan]Investigation Checklist[/bold cyan]")
        for i, it in enumerate(items):
            mark = "[green][x][/green]" if i <= completed_index else "[ ]"
            pr = getattr(it, 'priority', 0)
            imp = getattr(it, 'expected_impact', None)
            cat = getattr(it, 'category', None)
            meta = f"prio {pr}"
            if imp:
                meta += f", {imp}"
            if cat:
                meta += f", {cat}"
            console.print(f"  {mark} {it.goal}  ({meta})")

    def run(self, plan_n: int = 5):
        """Run the agent using the unified autonomous flow."""
        # Initialize run tracker
        if '/' in self.project_id or Path(self.project_id).exists():
            project_dir = Path(self.project_id).resolve()
        else:
            project_dir = get_project_dir(self.project_id)
        
        output_dir = project_dir / "agent_runs"
        output_dir.mkdir(exist_ok=True, parents=True)
        run_file = output_dir / f"run_{self.agent.agent_id}.json"
        
        self.run_tracker = RunTracker(run_file)
        
        # Set up token tracker
        token_tracker = get_token_tracker()
        token_tracker.reset()
        
        # Capture command line arguments
        command_args = sys.argv
        self.run_tracker.set_run_info(self.agent.agent_id, command_args)
        
        # Display configuration (omit context window; not available in unified client)
        # Get the actual models being used
        agent_model_info = "default"
        guidance_model_info = "default"
        
        if self.config and 'models' in self.config:
            # Get agent model
            if 'agent' in self.config['models']:
                agent_config = self.config['models']['agent']
                provider = agent_config.get('provider', 'unknown')
                model = agent_config.get('model', 'unknown')
                agent_model_info = f"{provider}/{model}"
            
            # Get guidance model
            if 'guidance' in self.config['models']:
                guidance_config = self.config['models']['guidance']
                provider = guidance_config.get('provider', 'unknown')
                model = guidance_config.get('model', 'unknown')
                guidance_model_info = f"{provider}/{model}"
        
        # Get context limit from config
        context_cfg = self.config.get('context', {}) if self.config else {}
        max_tokens = context_cfg.get('max_tokens', 128000)
        compression_threshold = context_cfg.get('compression_threshold', 0.75)
        
        config_text = (
            f"[bold cyan]AUTONOMOUS SECURITY AGENT[/bold cyan]\n"
            f"Project: [yellow]{self.project_id}[/yellow]\n"
            f"Agent Model: [magenta]{agent_model_info}[/magenta]\n"
            f"Guidance Model: [cyan]{guidance_model_info}[/cyan]\n"
            f"Max Iterations: [green]{self.agent.max_iterations}[/green]\n"
            f"Context Limit: [blue]{max_tokens:,} tokens[/blue] (compress at {int(compression_threshold*100)}%)"
        )
        if self.time_limit_minutes:
            config_text += f"\nTime Limit: [red]{self.time_limit_minutes} minutes[/red]"
        console.print(Panel.fit(config_text, border_style="cyan"))

        # Prepare a compact progress callback
        def progress_cb(info: dict):
            status = info.get('status', '')
            msg = info.get('message', '')
            it = info.get('iteration', 0)
            def _short(s: str, n: int = 200) -> str:
                try:
                    return (s[: n - 3] + '...') if isinstance(s, str) and len(s) > n else (s or '')
                except Exception:
                    return ''
            if status == 'decision':
                act = info.get('action', '-')
                reasoning = info.get('reasoning', '')
                params = info.get('parameters', {}) or {}
                
                # Special formatting for deep_think
                if act == 'deep_think':
                    console.print(f"\n[bold magenta]🧠 Iter {it}: Calling Deep Think Model[/bold magenta]")
                    if reasoning:
                        console.print(f"  [yellow]Reason:[/yellow] {reasoning}")
                else:
                    console.print(f"[cyan]Iter {it} decision:[/cyan] {act}")
                    if reasoning:
                        console.print(f"  [dim]Thought:[/dim] {reasoning}")  # Don't abbreviate thoughts
                    if params:
                        try:
                            import json as _json
                            console.print(f"  [dim]Params:[/dim] {_short(_json.dumps(params, separators=(',', ':')))}")
                        except Exception:
                            pass
            elif status == 'result':
                # Special handling for deep_think results
                action = info.get('action', '')
                result = info.get('result', {})
                
                if action == 'deep_think':
                    if result.get('status') == 'success':
                        console.print(f"\n[bold magenta]═══ DEEP THINK ANALYSIS ═══[/bold magenta]")
                        
                        # Show the compact response
                        full_response = result.get('full_response', '')
                        if full_response:
                            # Simply display the compact output
                            console.print(Panel(full_response, border_style="magenta"))
                        
                        # Show hypotheses formed count
                        hypotheses_formed = result.get('hypotheses_formed', 0)
                        if hypotheses_formed > 0:
                            console.print(f"[bold green]✓ Added {hypotheses_formed} hypothesis(es) to global store[/bold green]")
                        console.print()
                    else:
                        # Deep think failed - show the error
                        error_msg = result.get('error', 'Unknown error')
                        console.print(f"\n[bold red]❌ Deep Think Error:[/bold red] {error_msg}")
                        console.print("[yellow]Continuing with agent exploration...[/yellow]")
                else:
                    console.print(f"[dim]Iter {it} {status}:[/dim] {msg}")
            elif status in { 'analyzing', 'executing', 'hypothesis_formed' }:
                console.print(f"[dim]Iter {it} {status}:[/dim] {msg}")

        # Compose audit prompt
        audit_prompt = (
            "Perform a focused security audit of this codebase based on the available graphs. "
            "Identify potential vulnerabilities or risky patterns, form hypotheses, and summarize findings."
        )

        results = []
        planned_round = 0
        start_overall = time.time()
        while True:
            # Time limit check
            if self.time_limit_minutes:
                elapsed_minutes = (time.time() - start_overall) / 60.0
                if elapsed_minutes >= self.time_limit_minutes:
                    console.print(f"\n[yellow]⏰ Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                    break

            planned_round += 1
            console.print(f"\n[bold cyan]Planning batch {planned_round} (top {plan_n})[/bold cyan]")
            items = self._plan_investigations(max(1, plan_n))
            if not items:
                console.print("[green]No further promising investigations suggested — audit complete[/green]")
                break

            # Show previously completed investigations if any
            if self.completed_investigations:
                console.print("\n[dim]Previously completed investigations:[/dim]")
                for goal in self.completed_investigations:  # Show all completed
                    console.print(f"  [green]✓[/green] {goal}")
            
            # Only show new investigations in checklist
            console.print("\n[bold cyan]New Investigations Planned[/bold cyan]")
            for i, it in enumerate(items):
                pr = getattr(it, 'priority', 0)
                imp = getattr(it, 'expected_impact', None)
                cat = getattr(it, 'category', None)
                meta = f"prio {pr}"
                if imp:
                    meta += f", {imp}"
                if cat:
                    meta += f", {cat}"
                console.print(f"  [ ] {it.goal}  ({meta})")

            for idx, inv in enumerate(items):
                console.print(f"\n[bold blue]→ Investigating:[/bold blue] {inv.goal}")
                max_iters = self.agent.max_iterations if self.agent.max_iterations else 5
                self.start_time = time.time()
                try:
                    report = self.agent.investigate(inv.goal, max_iterations=max_iters, progress_callback=progress_cb)
                    results.append((inv, report))
                    # Track completed investigation
                    self.completed_investigations.append(inv.goal)
                    
                    # Update run tracker with investigation and token usage
                    self.run_tracker.add_investigation({
                        'goal': inv.goal,
                        'priority': getattr(inv, 'priority', 0),
                        'category': getattr(inv, 'category', None),
                        'iterations_completed': report.get('iterations_completed', 0) if report else 0,
                        'hypotheses': report.get('hypotheses', {}) if report else {}
                    })
                    self.run_tracker.update_token_usage(token_tracker.get_summary())
                except Exception as e:
                    self.run_tracker.add_error(str(e))
                    raise
                # Show progress
                console.print(f"[green]✓ Completed:[/green] {inv.goal}")

                # Early stop if agent is satisfied (no hypotheses and no more actions suggested)
                try:
                    hyp = (report or {}).get('hypotheses', {})
                    total_h = int(hyp.get('total', 0))
                except Exception:
                    total_h = 0
                if total_h == 0:
                    console.print("[dim]No hypotheses formed; considering coverage achieved for this thread[/dim]")
                    # Continue to next planned item; planner will run again next round
            # Loop to next planning round

        # After audit, show the last report in detail
        if results:
            last_report = results[-1][1]
            try:
                display_investigation_report(last_report)
            except Exception:
                console.print(f"\n[bold]Iterations:[/bold] {last_report.get('iterations_completed', 0)}")
                console.print(f"[bold]Hypotheses:[/bold] {last_report.get('hypotheses', {})}")

        # Finalize run tracker with final token usage
        self.run_tracker.update_token_usage(token_tracker.get_summary())
        self.run_tracker.finalize(status='completed')
        console.print(f"\n[green]Run details saved to:[/green] {run_file}")
    
    def _generate_enhanced_summary(self):
        """Deprecated in unified agent flow; retained for API compatibility."""
        return {
            'note': 'Use report returned by agent.investigate() for results',
        }
    
    def finalize_tracking(self, status: str = 'completed'):
        """Finalize run tracking with given status."""
        if self.run_tracker:
            token_tracker = get_token_tracker()
            self.run_tracker.update_token_usage(token_tracker.get_summary())
            self.run_tracker.finalize(status=status)


@click.command()
@click.argument('project_id')
@click.option('--iterations', type=int, help='Max iterations per investigation')
@click.option('--plan-n', type=int, default=5, help='Number of investigations to plan per batch (default: 5)')
@click.option('--time-limit', type=int, help='Time limit in minutes')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--resume', is_flag=True, help='Resume from previous session')
@click.option('--debug', is_flag=True, help='Enable debug logging of prompts and responses')
@click.option('--platform', default=None, help='Override LLM platform (e.g., openai, anthropic)')
@click.option('--model', default=None, help='Override LLM model (e.g., gpt-4, claude-3)')
def agent(project_id: str, iterations: Optional[int], plan_n: int, time_limit: Optional[int], 
          config: Optional[str], resume: bool, debug: bool, platform: Optional[str], model: Optional[str]):
    """Run autonomous security analysis agent."""
    
    if resume:
        console.print("[yellow]Resume functionality not yet implemented[/yellow]")
        return
    
    config_path = Path(config) if config else None
    
    runner = AgentRunner(project_id, config_path, iterations, time_limit, debug, platform, model)
    
    if not runner.initialize():
        return
    
    try:
        runner.run(plan_n=plan_n)
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent interrupted by user[/yellow]")
        # Try to save partial results
        try:
            runner.finalize_tracking('interrupted')
        except:
            pass
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        # Try to save partial results
        try:
            runner.finalize_tracking('failed')
        except:
            pass
        raise
