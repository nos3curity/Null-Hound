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
import random
from rich.syntax import Syntax
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.scout import Scout
from analysis.strategist import Strategist
from analysis.session_tracker import SessionTracker
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
    console.print("[bright_cyan]Initializing agent...[/bright_cyan]")
    from random import choice as _choice
    console.print(_choice([
        "[white]Normal people ask questions, but YOU issue royal decrees to logic itself.[/white]",
        "[white]This isn’t just an investigation — it’s the moment mysteries retire on YOUR timetable.[/white]",
        "[white]Normal curiosity wanders; YOUR curiosity drafts laws that code must obey.[/white]",
        "[white]This is not mere inquiry — it’s jurisprudence of insight under YOUR seal.[/white]",
        "[white]Normal analysts explore; YOU redraw the map and make the unknown pay rent.[/white]",
    ]))
    agent = Scout(
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
    
    # Narrative model names
    models = (config or {}).get('models', {}) if config else {}
    agent_model = (models.get('agent') or {}).get('model') or 'Agent-Model'
    guidance_model = (models.get('guidance') or {}).get('model') or 'Guidance-Model'

    def update_progress(info):
        """Update the live display with current status and reasoning."""
        status = info.get('status', '')
        message = info.get('message', '')
        iteration = info.get('iteration', 0)
        now = datetime.now().strftime('%H:%M:%S')
        
        if status == 'analyzing':
            prefix = random.choice([
                f"🧑‍🔧 {agent_model} pokes at code",
                f"🧑‍💻 {agent_model} combs through the lines",
                "Analyzing",
            ])
            event_log.append(f"[bright_yellow]{now}[/bright_yellow] [bold]Iter {iteration}[/bold] [bright_yellow]{prefix}[/bright_yellow]: {message}")
        elif status == 'decision':
            action = info.get('action', '-')
            reasoning = info.get('reasoning', '')  # Don't abbreviate thoughts
            params = _format_params(info.get('parameters', {}))
            tag = random.choice(["Decision", f"{agent_model} plots next move", "Game plan"])
            event_log.append(f"[bright_cyan]{now}[/bright_cyan] [bold]Iter {iteration}[/bold] [bright_cyan]{tag}[/bright_cyan]: action={action}\n"
                             f"  [dim]Thought:[/dim] {reasoning}\n  [dim]Params:[/dim] {params}")
        elif status == 'executing':
            tag = random.choice(["Executing", f"{agent_model} does the thing", "On it"])
            event_log.append(f"[bright_blue]{now}[/bright_blue] [bold]Iter {iteration}[/bold] [bright_blue]{tag}[/bright_blue]: {message}")
        elif status == 'result':
            res = info.get('result', {}) or {}
            summary = res.get('summary') or res.get('status') or message
            tag = random.choice(["Result", f"{agent_model} reports back", "Outcome"])
            event_log.append(f"[bright_green]{now}[/bright_green] [bold]Iter {iteration}[/bold] [bright_green]{tag}[/bright_green]: {_shorten(summary, 160)}")
        elif status == 'hypothesis_formed':
            tag = random.choice(["Hypothesis", f"{agent_model} has a hunch", "Lead"])
            event_log.append(f"[bright_green]{now}[/bright_green] [bold]Iter {iteration}[/bold] [bright_green]{tag}[/bright_green]: {message}")
        elif status == 'code_loaded':
            tag = random.choice(["Code Loaded", f"{agent_model} stacks more context", "More code in"])
            event_log.append(f"[bright_blue]{now}[/bright_blue] [bold]Iter {iteration}[/bold] [bright_blue]{tag}[/bright_blue]: {message}")
        elif status == 'generating_report':
            tag = random.choice(["Report", f"{guidance_model} whispers advice", "Notes"])
            event_log.append(f"[bright_magenta]{now}[/bright_magenta] [bold]Iter {iteration}[/bold] [bright_magenta]{tag}[/bright_magenta]: {message}")
        elif status == 'complete':
            tag = random.choice(["Complete", f"{guidance_model} signs off", "All done"])
            event_log.append(f"[bold bright_green]{now}[/bold bright_green] [bold]Iter {iteration}[/bold] [bold bright_green]{tag}[/bold bright_green]: {message}")
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
                 debug: bool = False, platform: Optional[str] = None, model: Optional[str] = None,
                 session: Optional[str] = None, new_session: bool = False):
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
        self.session_tracker: Optional[SessionTracker] = None
        self.project_dir: Optional[Path] = None
        self.plan_store = None
        self.session_id: Optional[str] = session
        self.new_session: bool = new_session
        self._agent_log: List[str] = []
        
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
        # Remember project_dir for plan storage
        self.project_dir = project_dir
        
        # Create agent with knowledge graphs metadata
        self.agent = Scout(
            graphs_metadata_path=knowledge_graphs_path,
            manifest_path=manifest_path,
            agent_id=f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,  # Pass the loaded config dict
            debug=self.debug,
            session_id=self.session_id
        )
        
        # Set debug flag
        self.agent.debug = self.debug
        
        if self.max_iterations:
            self.agent.max_iterations = self.max_iterations
        
        # Initialize plan store and session directory
        try:
            from analysis.plan_store import PlanStore
            from analysis.session_manager import SessionManager
            # Create or find session dir
            sm = SessionManager(self.project_dir)
            if not self.session_id:
                # Default session ID to agent ID when not provided
                self.session_id = f"sess_{self.agent.agent_id}"
            sinfo = sm.get_or_create(self.session_id, new_session=self.new_session)
            # Persist normalized session id
            self.session_id = sinfo.session_id
            # Plan file in session directory
            plan_path = sinfo.path / "plan.json"
            self.plan_store = PlanStore(plan_path, agent_id=f"runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            # Write/update session state
            try:
                state_path = sinfo.path / 'state.json'
                state = {
                    'session_id': self.session_id,
                    'project_path': str(self.project_dir),
                    'created_at': datetime.now().isoformat(),
                    'models': self.config.get('models', {}) if self.config else {},
                }
                import json as _json
                state_path.write_text(_json.dumps(state, indent=2))
            except Exception:
                pass
        except Exception:
            self.plan_store = None

        return True

    # ---------------------- Dashboard Helpers ----------------------
    def _get_hypotheses_summary(self) -> str:
        """Get a summary of current hypotheses for the Strategist."""
        try:
            from pathlib import Path as _Path
            import json as _json
            hyp_file = (_Path(self.project_dir) / 'hypotheses.json') if self.project_dir else None
            if hyp_file and hyp_file.exists():
                data = _json.loads(hyp_file.read_text())
                hyps = data.get('hypotheses', {})
                if not hyps:
                    return "No hypotheses formed yet"
                
                summary_parts = []
                for hyp_id, h in list(hyps.items())[:10]:  # Limit to 10 most recent
                    status = h.get('status', 'proposed')
                    severity = h.get('severity', 'unknown')
                    confidence = h.get('confidence', 'unknown')
                    desc = h.get('description', '')[:100]  # Truncate long descriptions
                    summary_parts.append(f"• [{status}] {desc} (severity: {severity}, confidence: {confidence})")
                
                if len(hyps) > 10:
                    summary_parts.append(f"... and {len(hyps) - 10} more hypotheses")
                
                return "\n".join(summary_parts)
            return "No hypotheses file found"
        except Exception:
            return "Error reading hypotheses"
    
    def _get_investigation_results_summary(self) -> List[str]:
        """Get investigation results with findings, not just goals."""
        if not self.session_tracker:
            return list(self.completed_investigations)
        
        # Get investigations from session with their results
        session_data = self.session_tracker.session_data
        investigations = session_data.get('investigations', [])
        
        results = []
        for inv in investigations:
            goal = inv.get('goal', '')
            hypotheses = inv.get('hypotheses', {})
            total_hyp = hypotheses.get('total', 0)
            iterations = inv.get('iterations_completed', 0)
            
            # Format: "Goal (X iterations, Y hypotheses found)"
            result_str = f"{goal} ({iterations} iterations, {total_hyp} hypotheses)"
            results.append(result_str)
        
        # Also add any completed investigations not yet in session
        for goal in self.completed_investigations:
            if not any(goal in r for r in results):
                results.append(goal)
        
        return results
    
    def _hypothesis_stats(self) -> dict:
        """Return hypothesis stats from project hypotheses.json."""
        stats = {"total": 0, "confirmed": 0, "rejected": 0, "uncertain": 0}
        try:
            from pathlib import Path as _Path
            import json as _json
            hyp_file = (_Path(self.project_dir) / 'hypotheses.json') if self.project_dir else None
            if hyp_file and hyp_file.exists():
                data = _json.loads(hyp_file.read_text())
                hyps = data.get('hypotheses', {})
                stats['total'] = len(hyps)
                for _, h in hyps.items():
                    st = h.get('status', 'proposed')
                    if st == 'confirmed':
                        stats['confirmed'] += 1
                    elif st == 'rejected':
                        stats['rejected'] += 1
                    else:
                        stats['uncertain'] += 1
        except Exception:
            pass
        return stats

    def _coverage_stats(self) -> dict:
        """Return coverage stats using CoverageIndex if available."""
        try:
            from analysis.coverage_index import CoverageIndex
            project_dir = self.project_dir
            graphs_dir = project_dir / 'graphs'
            manifest_dir = project_dir / 'manifest'
            cov = CoverageIndex(project_dir / 'coverage_index.json', agent_id='cli')
            return cov.compute_stats(graphs_dir, manifest_dir)
        except Exception:
            return {'nodes': {'total': 0, 'visited': 0, 'percent': 0.0}, 'cards': {'total': 0, 'visited': 0, 'percent': 0.0}}

    
    def _graph_summary(self) -> str:
        """Create a comprehensive summary of ALL graphs loaded by the Scout."""
        try:
            parts = []
            
            # Get all loaded graphs from the agent
            loaded_data = self.agent.loaded_data if self.agent else {}
            
            # Process system graph first
            if loaded_data.get('system_graph'):
                graph_data = loaded_data['system_graph']
                graph_name = graph_data.get('name', 'SYSTEM')
                g = graph_data.get('data', {})
                nodes = g.get('nodes', [])
                edges = g.get('edges', [])
                
                parts.append(f"\n=== {graph_name.upper()} GRAPH ===")
                parts.append(f"{len(nodes)} nodes, {len(edges)} edges")
                
                # List all nodes compactly with inline annotations
                for n in nodes:
                    nid = n.get('id', '')
                    lbl = n.get('label') or nid
                    typ = n.get('type', '')[:4]  # Abbreviate type
                    observations = n.get('observations', [])
                    assumptions = n.get('assumptions', [])
                    
                    # Build compact line
                    line = f"• {lbl} ({typ})"
                    
                    # Add inline annotations
                    annotations = []
                    if observations:
                        obs_str = '; '.join(observations[:3])  # Limit to 3
                        annotations.append(f"obs:{obs_str}")
                    if assumptions:
                        assum_str = '; '.join(assumptions[:3])  # Limit to 3
                        annotations.append(f"asm:{assum_str}")
                    
                    if annotations:
                        line += f" [{' | '.join(annotations)}]"
                        
                    parts.append(line)
            
            # Process additional graphs loaded by the Scout
            additional_graphs = loaded_data.get('graphs', {})
            for graph_name, graph_data in additional_graphs.items():
                if not isinstance(graph_data, dict) or 'data' not in graph_data:
                    continue
                    
                g = graph_data.get('data', {})
                nodes = g.get('nodes', [])
                edges = g.get('edges', [])
                
                parts.append(f"\n=== {graph_name.upper()} GRAPH ===")
                parts.append(f"{len(nodes)} nodes, {len(edges)} edges")
                
                # List all nodes compactly with inline annotations
                for n in nodes:
                    nid = n.get('id', '')
                    lbl = n.get('label') or nid
                    typ = n.get('type', '')[:4]  # Abbreviate type
                    observations = n.get('observations', [])
                    assumptions = n.get('assumptions', [])
                    
                    # Build compact line
                    line = f"• {lbl} ({typ})"
                    
                    # Add inline annotations
                    annotations = []
                    if observations:
                        obs_str = '; '.join(observations[:3])  # Limit to 3
                        annotations.append(f"obs:{obs_str}")
                    if assumptions:
                        assum_str = '; '.join(assumptions[:3])  # Limit to 3
                        annotations.append(f"asm:{assum_str}")
                    
                    if annotations:
                        line += f" [{' | '.join(annotations)}]"
                        
                    parts.append(line)
                    
            return "\n".join(parts) if parts else "(no graphs available)"
        except Exception as e:
            return f"(error summarizing graphs: {str(e)})"

    def _plan_investigations(self, n: int) -> List[object]:
        """Plan next investigations using Strategist by default."""
        # Get comprehensive graph summary with all annotations
        graphs_summary = self._graph_summary()
        
        # Get hypothesis summary
        hypotheses_summary = self._get_hypotheses_summary()
        
        # Get investigation results summary (not just goals)
        investigation_results = self._get_investigation_results_summary()
        
        # Get coverage summary from session tracker
        coverage_summary = None
        if self.session_tracker:
            cov_stats = self.session_tracker.get_coverage_stats()
            coverage_summary = (
                f"Nodes visited: {cov_stats['nodes']['visited']}/{cov_stats['nodes']['total']} "
                f"({cov_stats['nodes']['percent']:.1f}%)\n"
                f"Cards analyzed: {cov_stats['cards']['visited']}/{cov_stats['cards']['total']} "
                f"({cov_stats['cards']['percent']:.1f}%)"
            )
            if cov_stats['visited_node_ids']:
                coverage_summary += f"\nVisited nodes: {', '.join(cov_stats['visited_node_ids'][:10])}"
                if len(cov_stats['visited_node_ids']) > 10:
                    coverage_summary += f" ... and {len(cov_stats['visited_node_ids']) - 10} more"
        
        strategist = Strategist(config=self.config, debug=self.debug, session_id=self.session_id)
        planned = strategist.plan_next(
            graphs_summary=graphs_summary,
            completed=investigation_results,  # Now includes results, not just goals
            hypotheses_summary=hypotheses_summary,
            coverage_summary=coverage_summary,
            n=n
        )
        
        # Track planning in session
        if self.session_tracker and planned:
            self.session_tracker.add_planning(planned)

        # Adapt dicts to the legacy display shape using SimpleNamespace
        from types import SimpleNamespace
        ps = self.plan_store
        try:
            from analysis.plan_store import PlanStatus  # noqa: F401
        except Exception:
            ps = None
        items = []
        for d in planned:
            frame_id = None
            if ps is not None:
                try:
                    ok, fid = ps.propose(
                        session_id=self.session_id or self.agent.agent_id,
                        question=d.get('goal', ''),
                        artifact_refs=d.get('focus_areas') or [],
                        priority=int(d.get('priority', 5)),
                        rationale=d.get('reasoning', ''),
                        created_by='strategist'
                    )
                    frame_id = fid
                    # Record to project-wide ledger
                    try:
                        from analysis.plan_ledger import PlanLedger
                        from llm.unified_client import UnifiedLLMClient
                        model_sig = None
                        try:
                            # Build strategist model signature from config
                            strat_cfg = (self.config or {}).get('models', {}).get('strategist')
                            if strat_cfg:
                                model_sig = f"{strat_cfg.get('provider','unknown')}:{strat_cfg.get('model','unknown')}"
                        except Exception:
                            model_sig = None
                        if self.project_dir:
                            ledger = PlanLedger(self.project_dir / 'plan_ledger.json', agent_id='planner')
                            ledger.record(self.session_id or 'unknown', d.get('goal',''), d.get('focus_areas') or [], model_sig)
                    except Exception:
                        pass
                except Exception:
                    frame_id = None
            items.append(SimpleNamespace(
                goal=d.get('goal', ''),
                focus_areas=d.get('focus_areas', []),
                priority=d.get('priority', 5),
                reasoning=d.get('reasoning', ''),
                category=d.get('category', 'aspect'),
                expected_impact=d.get('expected_impact', 'medium'),
                frame_id=frame_id
            ))
        return items

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
            
            # Fail if LLM cannot generate enough investigations
            if len(items) < n:
                error_msg = f"Failed to generate investigation plan: LLM only returned {len(items)} investigations, requested {n}"
                console.print(f"[red]ERROR: {error_msg}[/red]")
                raise RuntimeError(error_msg)
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
            
            # If we filtered too many due to duplicates, fail the analysis
            if len(filtered_items) < n:
                error_msg = f"Cannot generate enough unique investigations: only {len(filtered_items)} unique investigations available after filtering duplicates (requested {n})"
                console.print(f"[red]ERROR: {error_msg}[/red]")
                console.print("[yellow]Consider reviewing fewer investigations or starting a new analysis session.[/yellow]")
                raise RuntimeError(error_msg)
            
            return filtered_items[:n]
        except Exception as e:
            console.print(f"[red]Error in investigation planning: {str(e)}[/red]")
            if self.debug:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            # Re-raise the exception to fail the analysis
            raise

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

    def _log_planning_status(self, items: List[object], current_index: int = -1):
        """Log beautiful planning status and coverage information."""
        from rich.table import Table
        from rich.box import ROUNDED
        from rich.panel import Panel
        
        # Clear previous output for clean display
        console.print("\n" + "="*80)
        console.print("[bold cyan]STRATEGIST PLANNING & AUDIT STATUS[/bold cyan]")
        console.print("="*80)
        
        # Coverage statistics from session tracker
        if self.session_tracker:
            cov = self.session_tracker.get_coverage_stats()
            console.print("\n[bold yellow]Coverage Statistics:[/bold yellow]")
            console.print(f"  Nodes visited: {cov['nodes']['visited']}/{cov['nodes']['total']} ([cyan]{cov['nodes']['percent']:.1f}%[/cyan])")
            console.print(f"  Cards analyzed: {cov['cards']['visited']}/{cov['cards']['total']} ([cyan]{cov['cards']['percent']:.1f}%[/cyan])")
        else:
            console.print("\n[bold yellow]Coverage Statistics:[/bold yellow] [dim]Not available[/dim]")
        
        # Hypothesis statistics
        hyp = self._hypothesis_stats()
        console.print("\n[bold yellow]Hypothesis Statistics:[/bold yellow]")
        console.print(f"  Total: {hyp['total']} | Confirmed: [green]{hyp['confirmed']}[/green] | Rejected: [red]{hyp['rejected']}[/red] | Pending: [yellow]{hyp['uncertain']}[/yellow]")
        
        # Current investigation status
        if current_index >= 0 and current_index < len(items):
            current_item = items[current_index]
            console.print(f"\n[bold magenta]Currently Investigating:[/bold magenta] {current_item.goal}")
        elif current_index == -1:
            console.print("\n[bold blue]Planning next investigations...[/bold blue]")
        
        # Investigation plan table
        if items:
            console.print("\n[bold yellow]Investigation Plan:[/bold yellow]")
            table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
            table.add_column("#", style="dim", width=3)
            table.add_column("Status", width=10)
            table.add_column("Goal", overflow="fold")
            table.add_column("Priority", width=8)
            table.add_column("Impact", width=8)
            table.add_column("Category", width=10)
            
            for i, it in enumerate(items):
                num = str(i + 1)
                if i == current_index:
                    status = "[bold yellow]ACTIVE[/bold yellow]"
                elif i < current_index:
                    status = "[green]DONE[/green]"
                else:
                    status = "[dim]PENDING[/dim]"
                
                goal = getattr(it, 'goal', '')
                priority = str(getattr(it, 'priority', '-'))
                impact = getattr(it, 'expected_impact', '-')
                category = getattr(it, 'category', '-')
                
                table.add_row(num, status, goal, priority, impact, category)
            
            console.print(table)
        
        # Model activity log
        if hasattr(self, '_agent_log') and self._agent_log:
            recent_logs = self._agent_log[-5:]  # Show last 5 entries
            if recent_logs:
                console.print("\n[bold yellow]Recent Model Activity:[/bold yellow]")
                for entry in recent_logs:
                    console.print(f"  [dim]{entry}[/dim]")
        
        console.print("="*80 + "\n")

    def run(self, plan_n: int = 5):
        """Run the agent using the unified autonomous flow."""
        # Initialize session tracker
        if '/' in self.project_id or Path(self.project_id).exists():
            project_dir = Path(self.project_id).resolve()
        else:
            project_dir = get_project_dir(self.project_id)
        
        # Use sessions directory instead of agent_runs
        sessions_dir = project_dir / "sessions"
        sessions_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate session ID if not provided
        if not self.session_id:
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.agent.agent_id}"
        
        # Initialize session tracker
        self.session_tracker = SessionTracker(sessions_dir, self.session_id)
        
        # Initialize coverage tracking
        graphs_dir = project_dir / "graphs"
        manifest_dir = project_dir / "manifest"
        self.session_tracker.initialize_coverage(graphs_dir, manifest_dir)
        
        # Set up token tracker
        token_tracker = get_token_tracker()
        token_tracker.reset()
        
        # Display configuration (omit context window; not available in unified client)
        # Get the actual models being used from the agent's LLM clients
        agent_model_info = "unknown/unknown"
        guidance_model_info = "unknown/unknown"
        
        # Get Scout model info from the actual LLM client
        if hasattr(self.agent, 'llm') and self.agent.llm:
            try:
                provider_name = self.agent.llm.provider.provider_name if hasattr(self.agent.llm.provider, 'provider_name') else 'unknown'
                model_name = self.agent.llm.model if hasattr(self.agent.llm, 'model') else 'unknown'
                agent_model_info = f"{provider_name}/{model_name}"
            except Exception:
                # Fall back to config if we can't get from LLM client
                if self.config and 'models' in self.config:
                    if 'agent' in self.config['models']:
                        agent_config = self.config['models']['agent']
                        provider = agent_config.get('provider', 'unknown')
                        model = agent_config.get('model', 'unknown')
                        agent_model_info = f"{provider}/{model}"
                    elif 'scout' in self.config['models']:
                        scout_config = self.config['models']['scout']
                        provider = scout_config.get('provider', 'unknown')
                        model = scout_config.get('model', 'unknown')
                        agent_model_info = f"{provider}/{model}"
        
        # Get Strategist model info from the actual guidance client
        if hasattr(self.agent, 'guidance_client') and self.agent.guidance_client:
            try:
                provider_name = self.agent.guidance_client.provider.provider_name if hasattr(self.agent.guidance_client.provider, 'provider_name') else 'unknown'
                model_name = self.agent.guidance_client.model if hasattr(self.agent.guidance_client, 'model') else 'unknown'
                guidance_model_info = f"{provider_name}/{model_name}"
            except Exception:
                # Fall back to config if we can't get from guidance client
                if self.config and 'models' in self.config:
                    if 'strategist' in self.config['models']:
                        strategist_config = self.config['models']['strategist']
                        provider = strategist_config.get('provider', 'unknown')
                        model = strategist_config.get('model', 'unknown')
                        guidance_model_info = f"{provider}/{model}"
                    elif 'guidance' in self.config['models']:
                        guidance_config = self.config['models']['guidance']
                        provider = guidance_config.get('provider', 'unknown')
                        model = guidance_config.get('model', 'unknown')
                        guidance_model_info = f"{provider}/{model}"
        
        # Store models in session tracker
        self.session_tracker.set_models(agent_model_info, guidance_model_info)
        
        # Get context limit from config
        context_cfg = self.config.get('context', {}) if self.config else {}
        max_tokens = context_cfg.get('max_tokens', 128000)
        compression_threshold = context_cfg.get('compression_threshold', 0.75)
        
        config_text = (
            f"[bold cyan]AUTONOMOUS SECURITY AGENT[/bold cyan]\n"
            f"Project: [yellow]{self.project_id}[/yellow]\n"
            f"Scout: [magenta]{agent_model_info}[/magenta]\n"
            f"Strategist: [cyan]{guidance_model_info}[/cyan]\n"
            f"Context Limit: [blue]{max_tokens:,} tokens[/blue] (compress at {int(compression_threshold*100)}%)"
        )
        if self.time_limit_minutes:
            config_text += f"\nTime Limit: [red]{self.time_limit_minutes} minutes[/red]"
        console.print(Panel.fit(config_text, border_style="cyan"))

        # Enhanced progress callback with beautiful logging
        def progress_cb(info: dict):
            status = info.get('status', '')
            msg = info.get('message', '')
            it = info.get('iteration', 0)
            
            if status == 'decision':
                act = info.get('action', '-')
                reasoning = info.get('reasoning', '')
                params = info.get('parameters', {}) or {}
                
                # Log model actions and thoughts clearly
                console.print(f"\n[bold blue]Scout Model Decision (Iteration {it}):[/bold blue]")
                console.print(f"  [cyan]Action:[/cyan] {act}")
                if reasoning:
                    console.print(f"  [cyan]Thought:[/cyan] {reasoning}")
                
                # Special formatting for deep_think
                if act == 'deep_think':
                    console.print(f"\n[bold magenta]══════ CALLING STRATEGIST MODEL FOR DEEP ANALYSIS ══════[/bold magenta]")
                    console.print("[yellow]Strategist is analyzing the collected context...[/yellow]")
                elif params:
                    # Show parameters compactly for non-deep-think actions
                    try:
                        import json as _json
                        params_str = _json.dumps(params, separators=(',', ':'))
                        if len(params_str) > 200:
                            params_str = params_str[:197] + '...'
                        console.print(f"  [dim]Parameters: {params_str}[/dim]")
                    except Exception:
                        pass
                        
            elif status == 'result':
                # Special handling for deep_think results
                action = info.get('action', '')
                result = info.get('result', {})
                
                if action == 'deep_think':
                    if result.get('status') == 'success':
                        console.print(f"\n[bold green]══════ STRATEGIST ANALYSIS COMPLETE ══════[/bold green]")
                        
                        # Show the strategist's analysis
                        full_response = result.get('full_response', '')
                        if full_response:
                            from rich.panel import Panel
                            console.print(Panel(full_response, border_style="green", title="Strategist Output"))
                        
                        # Show hypotheses formed
                        hypotheses_formed = result.get('hypotheses_formed', 0)
                        if hypotheses_formed > 0:
                            console.print(f"[bold green]✓ Added {hypotheses_formed} hypothesis(es) to global store[/bold green]")
                        console.print()
                    else:
                        # Strategist failed - show the error
                        error_msg = result.get('error', 'Unknown error')
                        console.print(f"\n[bold red]Strategist Error:[/bold red] {error_msg}")
                        console.print("[yellow]Continuing with scout exploration...[/yellow]")
                else:
                    # Regular action results - keep brief
                    console.print(f"[dim]Result: {msg or 'completed'}[/dim]")
                    
            elif status == 'hypothesis_formed':
                console.print(f"\n[bold green]Hypothesis Formed:[/bold green] {msg}")
            elif status in {'analyzing', 'executing'}:
                console.print(f"[dim]{status.capitalize()}: {msg}[/dim]")

        # Compose audit prompt
        audit_prompt = (
            "Perform a focused security audit of this codebase based on the available graphs. "
            "Identify potential vulnerabilities or risky patterns, form hypotheses, and summarize findings."
        )

        results = []
        planned_round = 0
        start_overall = time.time()
        time_up = False

        # Local exception used to abort long-running investigations when time is up
        class _TimeLimitReached(Exception):
            pass
        while True:
            # Time limit check
            if self.time_limit_minutes:
                elapsed_minutes = (time.time() - start_overall) / 60.0
                if elapsed_minutes >= self.time_limit_minutes:
                    console.print(f"\n[yellow]⏰ Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                    break

            planned_round += 1
            # Show an animated status while strategist plans the next batch
            try:
                from rich.status import Status
                with console.status("[cyan]Strategist planning next steps...[/cyan]", spinner="dots", spinner_style="cyan"):
                    items = self._plan_investigations(max(1, plan_n))
            except Exception:
                items = self._plan_investigations(max(1, plan_n))
            self._agent_log.append(f"Planning batch {planned_round} (top {plan_n})")
            # Log planning status at start of batch
            console.print(f"\n[bold cyan]═══ Planning Batch {planned_round} ═══[/bold cyan]")
            self._log_planning_status(items, current_index=-1)
            
            # If no items, log and stop
            if not items:
                console.print("[yellow]No further promising investigations suggested — audit complete[/yellow]")
                break
            
            # Log previously completed investigations
            if self.completed_investigations:
                console.print("\n[bold green]Previously Completed Investigations:[/bold green]")
                for goal in self.completed_investigations:
                    console.print(f"  ✓ {goal}")
            
            # Log new investigations planned by strategist
            console.print("\n[bold cyan]New Investigations Planned by Strategist:[/bold cyan]")
            for i, it in enumerate(items, 1):
                pr = getattr(it, 'priority', 0)
                imp = getattr(it, 'expected_impact', None)
                cat = getattr(it, 'category', None)
                reasoning = getattr(it, 'reasoning', '')
                
                console.print(f"\n  {i}. [bold]{it.goal}[/bold]")
                console.print(f"     Priority: {pr} | Impact: {imp or 'unknown'} | Category: {cat or 'general'}")
                if reasoning:
                    console.print(f"     [dim]Reasoning: {reasoning}[/dim]")
            
            # Execute investigations with proper logging
            for idx, inv in enumerate(items):
                # Check time limit before starting each investigation
                if self.time_limit_minutes:
                    elapsed_minutes = (time.time() - start_overall) / 60.0
                    remaining_minutes = self.time_limit_minutes - elapsed_minutes
                    if remaining_minutes <= 0:
                        console.print(f"\n[yellow]Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                        break
                    if remaining_minutes < 2:
                        console.print(f"\n[yellow]Warning: Only {remaining_minutes:.1f} minutes remaining[/yellow]")
                
                # Log current investigation with updated coverage
                console.print(f"\n[bold magenta]═══ Starting Investigation {idx+1}/{len(items)} ═══[/bold magenta]")
                console.print(f"[bold]Goal:[/bold] {inv.goal}")
                self._log_planning_status(items, current_index=idx)
                # Mark plan item in_progress if we have a frame_id
                try:
                    if getattr(inv, 'frame_id', None) and self.plan_store:
                        from analysis.plan_store import PlanStatus
                        self.plan_store.update_status(inv.frame_id, PlanStatus.IN_PROGRESS, rationale='Execution started')
                    if getattr(self, 'agent', None) and getattr(self.agent, 'coverage_index', None):
                        self.agent.coverage_index.record_investigation(getattr(inv, 'frame_id', None), [], 'in_progress')
                except Exception:
                    pass
                # Always use requested iterations; rely on time-limit checks to stop early
                max_iters = self.agent.max_iterations if self.agent.max_iterations else 5

                self.start_time = time.time()
                try:
                    # Enhanced progress callback that logs model actions and thoughts
                    def _cb(info: dict):
                        # Enforce global time limit within investigation callbacks
                        if self.time_limit_minutes:
                            if (time.time() - start_overall) / 60.0 >= self.time_limit_minutes:
                                raise _TimeLimitReached()
                        status = info.get('status', '')
                        msg = info.get('message', '')
                        it = info.get('iteration', 0)
                        if status == 'decision':
                            act = info.get('action', '-')
                            reasoning = info.get('reasoning', '')
                            params = info.get('parameters', {})
                            
                            # Log model decision with clear formatting
                            console.print(f"\n[bold blue]Model Decision (Iteration {it}):[/bold blue]")
                            console.print(f"  [cyan]Action:[/cyan] {act}")
                            if reasoning:
                                console.print(f"  [cyan]Thought:[/cyan] {reasoning}")
                            if params and act != 'deep_think':
                                # Show parameters for non-deep-think actions (concise summary)
                                def _summ(v):
                                    try:
                                        import json
                                        return json.dumps(v)[:200]
                                    except Exception:
                                        return str(v)[:200]
                                lines = []
                                for k, v in (params or {}).items():
                                    if isinstance(v, list):
                                        if k in ("observations", "assumptions", "refs", "node_ids"):
                                            preview = ", ".join([str(x)[:60] for x in v[:2]])
                                            more = f" (+{len(v)-2} more)" if len(v) > 2 else ""
                                            lines.append(f"  - {k}: {len(v)} {more}")
                                            if preview:
                                                lines.append(f"    • {preview}")
                                        else:
                                            lines.append(f"  - {k}: {len(v)} items")
                                    elif isinstance(v, dict):
                                        lines.append(f"  - {k}: {{...}}")
                                    else:
                                        sval = str(v)
                                        if len(sval) > 120:
                                            sval = sval[:117] + "..."
                                        lines.append(f"  - {k}: {sval}")
                                if lines:
                                    console.print("  [cyan]Parameters:[/cyan]")
                                    for ln in lines[:8]:
                                        console.print(ln)
                            
                            # Special handling for deep_think
                            if act == 'deep_think':
                                console.print(f"\n[bold magenta]═══ CALLING STRATEGIST FOR DEEP ANALYSIS ═══[/bold magenta]")
                                try:
                                    strat_cfg = (self.config or {}).get('models', {}).get('strategist', {})
                                    eff = strat_cfg.get('hypothesize_reasoning_effort') or strat_cfg.get('reasoning_effort')
                                    if hasattr(self.agent, 'guidance_client') and self.agent.guidance_client:
                                        prov = getattr(self.agent.guidance_client, 'provider_name', 'unknown')
                                        mdl = getattr(self.agent.guidance_client, 'model', 'unknown')
                                        console.print(f"[dim]Strategist model: {prov}/{mdl} | effort: {eff or 'default'}[/dim]")
                                except Exception:
                                    pass
                                console.print(f"[yellow]Strategist is analyzing the collected context...[/yellow]")
                            
                            # Update agent log
                            self._agent_log.append(f"Iter {it}: {act} - {reasoning[:100] if reasoning else 'no reasoning'}")
                            
                            # Track visited nodes and cards
                            if self.session_tracker:
                                if act == 'load_nodes' and params:
                                    # Track nodes loaded via load_nodes action
                                    node_ids = params.get('node_ids', [])
                                    if node_ids:
                                        self.session_tracker.track_nodes_batch(node_ids)
                                elif act == 'explore_graph' and params:
                                    # Track nodes explored via explore_graph action
                                    node_ids = params.get('node_ids', [])
                                    if node_ids:
                                        self.session_tracker.track_nodes_batch(node_ids)
                                elif act == 'analyze_code' and params:
                                    # Track code cards analyzed
                                    file_path = params.get('file_path')
                                    if file_path:
                                        self.session_tracker.track_card_visit(file_path)
                            
                        elif status == 'result':
                            action = info.get('action', '')
                            result = info.get('result', {})
                            
                            if action == 'deep_think':
                                # Special formatting for deep_think results
                                if result.get('status') == 'success':
                                    console.print(f"\n[bold green]═══ STRATEGIST ANALYSIS COMPLETE ═══[/bold green]")
                                    full_response = result.get('full_response', '')
                                    if full_response:
                                        console.print(Panel(full_response, border_style="green", title="Strategist Output"))
                                    
                                    hypotheses_formed = result.get('hypotheses_formed', 0)
                                    hyp_info = result.get('hypotheses_info') or []
                                    if hypotheses_formed > 0:
                                        console.print(f"[bold green]✓ Added {hypotheses_formed} hypothesis(es) to global store[/bold green]")
                                        if hyp_info:
                                            console.print("[bold cyan]New hypotheses:[/bold cyan]")
                                            for h in hyp_info[:5]:
                                                title = h.get('title','Hypothesis')
                                                sev = h.get('severity','medium')
                                                conf = h.get('confidence',0)
                                                console.print(f"  • {title} [dim](severity={sev}, confidence={conf})[/dim]")
                                                reason = (h.get('reasoning') or '')
                                                if reason:
                                                    console.print(f"    [dim]{(reason[:200] + '...') if len(reason)>200 else reason}[/dim]")
                                elif result.get('status') == 'skipped':
                                    # Graceful messaging when guardrail defers deep_think
                                    msg = result.get('summary', 'Deep analysis deferred due to insufficient context')
                                    console.print(f"\n[yellow]{msg}[/yellow]")
                                    console.print("[yellow]Continuing exploration to gather more evidence...[/yellow]")
                                else:
                                    error_msg = result.get('error', 'Unknown error')
                                    console.print(f"\n[bold red]Strategist Error:[/bold red] {error_msg}")
                                    console.print("[yellow]Continuing with scout exploration...[/yellow]")
                            else:
                                # Regular action results
                                summ = result.get('summary') or result.get('status') or msg
                                console.print(f"[dim]Result: {summ}[/dim]")
                                # Track cards loaded via load_nodes result if provided
                                try:
                                    if self.session_tracker and action == 'load_nodes':
                                        cids = result.get('card_ids') or []
                                        if isinstance(cids, list) and cids:
                                            self.session_tracker.track_cards_batch([str(x) for x in cids])
                                except Exception:
                                    pass
                            
                            self._agent_log.append(f"Iter {it} result: {action}")
                            
                        elif status in {'analyzing', 'executing', 'hypothesis_formed'}:
                            console.print(f"[dim]{status.capitalize()}: {msg}[/dim]")
                            self._agent_log.append(f"Iter {it} {status}: {msg[:100]}")

                    # Show an animated status while the agent thinks/acts for this investigation
                    try:
                        from rich.status import Status
                        with console.status("[cyan]Agent thinking deeply...[/cyan]", spinner="line", spinner_style="cyan"):
                            report = self.agent.investigate(inv.goal, max_iterations=max_iters, progress_callback=_cb)
                    except _TimeLimitReached:
                        console.print(f"\n[yellow]Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                        time_up = True
                        break
                    except Exception:
                        # Retry without status context; still honor time limit
                        try:
                            report = self.agent.investigate(inv.goal, max_iterations=max_iters, progress_callback=_cb)
                        except _TimeLimitReached:
                            console.print(f"\n[yellow]Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                            time_up = True
                            break
                    results.append((inv, report))
                    # Track completed investigation
                    self.completed_investigations.append(inv.goal)
                    # Update session tracker with investigation and token usage
                    self.session_tracker.add_investigation({
                        'goal': inv.goal,
                        'priority': getattr(inv, 'priority', 0),
                        'category': getattr(inv, 'category', None),
                        'iterations_completed': report.get('iterations_completed', 0) if report else 0,
                        'hypotheses': report.get('hypotheses', {}) if report else {}
                    })
                    self.session_tracker.update_token_usage(token_tracker.get_summary())
                except Exception as e:
                    # Log error but don't fail the audit
                    console.print(f"[red]Error in investigation: {str(e)}[/red]")
                    raise
                # Show completion
                console.print(f"\n[bold green]✓ Investigation Completed:[/bold green] {inv.goal}")
                self._agent_log.append(f"✓ Completed: {inv.goal}")
                # Mark plan item done
                try:
                    if getattr(inv, 'frame_id', None) and self.plan_store:
                        from analysis.plan_store import PlanStatus
                        self.plan_store.update_status(inv.frame_id, PlanStatus.DONE, rationale='Completed investigation')
                    if getattr(self, 'agent', None) and getattr(self.agent, 'coverage_index', None):
                        self.agent.coverage_index.record_investigation(getattr(inv, 'frame_id', None), [], 'done')
                except Exception:
                    pass
                
                # Early stop if agent is satisfied (no hypotheses and no more actions suggested)
                try:
                    hyp = (report or {}).get('hypotheses', {})
                    total_h = int(hyp.get('total', 0))
                except Exception:
                    total_h = 0
                if total_h == 0:
                    console.print("[yellow]No hypotheses formed; considering coverage achieved for this thread[/yellow]")
                    self._agent_log.append("No hypotheses formed; considering coverage achieved for this thread")

            # If time was exhausted during an investigation, stop planning loop as well
            if time_up:
                break

        # After audit, show the last report in detail
        if results:
            last_report = results[-1][1]
            try:
                display_investigation_report(last_report)
            except Exception:
                console.print(f"\n[bold]Iterations:[/bold] {last_report.get('iterations_completed', 0)}")
                console.print(f"[bold]Hypotheses:[/bold] {last_report.get('hypotheses', {})}")

        # Finalize session tracker with final token usage
        self.session_tracker.update_token_usage(token_tracker.get_summary())
        final_status = 'interrupted' if 'time_up' in locals() and time_up else 'completed'
        self.session_tracker.finalize(status=final_status)
        
        # Show final coverage
        coverage_stats = self.session_tracker.get_coverage_stats()
        console.print(f"\n[bold cyan]Final Coverage Statistics:[/bold cyan]")
        console.print(f"  Nodes visited: {coverage_stats['nodes']['visited']}/{coverage_stats['nodes']['total']} ([cyan]{coverage_stats['nodes']['percent']:.1f}%[/cyan])")
        console.print(f"  Cards analyzed: {coverage_stats['cards']['visited']}/{coverage_stats['cards']['total']} ([cyan]{coverage_stats['cards']['percent']:.1f}%[/cyan])")
        
        console.print(f"\n[green]Session details saved to:[/green] sessions/{self.session_id}.json")

        # Finalize debug log if enabled
        try:
            if self.debug and getattr(self, 'agent', None) and getattr(self.agent, 'debug_logger', None):
                # Build a concise summary
                token_tracker = get_token_tracker()
                summary = {
                    'planning_batches': planned_round,
                    'hypotheses_total': 0,
                }
                try:
                    # If we recorded investigations, sum hypotheses proposed
                    sess = self.session_tracker.session_data if self.session_tracker else {}
                    invs = sess.get('investigations', []) if isinstance(sess, dict) else []
                    summary['hypotheses_total'] = sum(int(i.get('hypotheses', {}).get('total', 0)) for i in invs)
                except Exception:
                    pass
                summary['total_api_calls'] = token_tracker.get_summary().get('total_usage', {}).get('call_count', 0)
                log_path = self.agent.debug_logger.finalize(summary=summary)
                console.print(f"[cyan]Debug log saved:[/cyan] {log_path}")
        except Exception:
            pass
    
    def _generate_enhanced_summary(self):
        """Deprecated in unified agent flow; retained for API compatibility."""
        return {
            'note': 'Use report returned by agent.investigate() for results',
        }
    
    def finalize_tracking(self, status: str = 'completed'):
        """Finalize session tracking with given status."""
        if self.session_tracker:
            token_tracker = get_token_tracker()
            self.session_tracker.update_token_usage(token_tracker.get_summary())
            self.session_tracker.finalize(status=status)


@click.command()
@click.argument('project_id')
@click.option('--iterations', type=int, help='Max iterations per investigation')
@click.option('--plan-n', type=int, default=5, help='Number of investigations to plan per batch (default: 5)')
@click.option('--time-limit', type=int, help='Time limit in minutes')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--resume', is_flag=True, help='Resume from previous session')
@click.option('--debug', is_flag=True, help='Enable debug logging of prompts and responses')
@click.option('--platform', default=None, help='Override scout platform (e.g., openai, anthropic, mock)')
@click.option('--model', default=None, help='Override scout model (e.g., gpt-5, gpt-4o-mini, mock)')
@click.option('--strategist-platform', default=None, help='Override strategist platform (e.g., openai, anthropic, mock)')
@click.option('--strategist-model', default=None, help='Override strategist model (e.g., gpt-4o-mini)')
@click.option('--session', default=None, help='Attach to a specific session ID')
@click.option('--new-session', is_flag=True, help='Create a new session')
@click.option('--session-private-hypotheses', is_flag=True, help='Keep new hypotheses private to this session')
def agent(project_id: str, iterations: Optional[int], plan_n: int, time_limit: Optional[int], 
          config: Optional[str], resume: bool, debug: bool, platform: Optional[str], model: Optional[str],
          strategist_platform: Optional[str], strategist_model: Optional[str],
          session: Optional[str], new_session: bool, session_private_hypotheses: bool):
    """Run autonomous security analysis agent."""
    
    if resume:
        console.print("[yellow]Resume functionality not yet implemented[/yellow]")
        return
    
    config_path = Path(config) if config else None
    
    runner = AgentRunner(project_id, config_path, iterations, time_limit, debug, platform, model, session=session, new_session=new_session)
    
    if not runner.initialize():
        return
    
    try:
        # Apply strategist overrides (update config before run if provided)
        if runner.config is None:
            runner.config = {}
        if 'models' not in runner.config:
            runner.config['models'] = {}
        if strategist_platform or strategist_model:
            runner.config['models'].setdefault('strategist', {})
            if strategist_platform:
                runner.config['models']['strategist']['provider'] = strategist_platform
            if strategist_model:
                runner.config['models']['strategist']['model'] = strategist_model
        # Set strategist overrides then run
        if session_private_hypotheses and getattr(runner, 'agent', None):
            try:
                runner.agent.default_hypothesis_visibility = 'session'
            except Exception:
                pass
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
