#!/usr/bin/env python3
"""Graph building commands for Hound CLI."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import random

from analysis.debug_logger import DebugLogger
from analysis.graph_builder import GraphBuilder
from ingest.bundles import AdaptiveBundler
from ingest.manifest import RepositoryManifest
from llm.token_tracker import get_token_tracker
from visualization.dynamic_graph_viz import generate_dynamic_visualization

console = Console()
# Progress console writes to stderr; auto-detect TTY so interactive shells
# show progress bars, while non-TTY (benchmarks/pipes) suppress animations.
progress_console = Console(file=sys.stderr)


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file."""
    from utils.config_loader import load_config as _load_config
    config = _load_config(config_path)
    
    if not config and config_path:
        # Only error if a specific path was requested but not found
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        sys.exit(1)
    
    return config


def build(
    repo_path: str = typer.Argument(..., help="Path to repository to analyze"),
    repo_id: str | None = typer.Option(None, help="Repository ID"),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    max_iterations: int = typer.Option(3, "--iterations", "-i", help="Maximum iterations"),
    max_graphs: int = typer.Option(2, "--graphs", "-g", help="Number of graphs"),
    focus_areas: str | None = typer.Option(None, "--focus", "-f", help="Focus areas"),
    file_filter: str | None = typer.Option(None, "--files", help="File filter"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Generate HTML"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output and disable animations"),
):
    """Build agent-driven knowledge graphs."""
    time.time()
    config = load_config(config_path)
    
    repo_path = Path(repo_path).resolve()
    repo_name = repo_id or repo_path.name
    output_dir = Path(output_dir) if output_dir else Path(".hound_cache") / repo_name
    manifest_dir = output_dir / "manifest"
    graphs_dir = output_dir / "graphs"
    
    # Set up token tracker
    token_tracker = get_token_tracker()
    token_tracker.reset()
    
    # Header
    console.print(Panel.fit(
        f"[bold bright_cyan]Building Knowledge Graphs[/bold bright_cyan]\n"
        f"Repository: [white]{repo_path.name}[/white]\n"
        f"Graphs: [white]{max_graphs}[/white] | Iterations: [white]{max_iterations}[/white]",
        box=box.ROUNDED
    ))
    # A little hype for the journey
    from random import choice
    console.print(choice([
        "[white]Normal folks build graphs, but YOU draft constellations and make causality salute.[/white]",
        "[white]This isn’t just graph building — it’s YOU engraving laws of structure into the codebase.[/white]",
        "[white]Normal structure emerges; YOUR structure recruits the universe as documentation.[/white]",
        "[white]This is not just a graph — it’s a starmap and YOU hold the pen.[/white]",
        "[white]Normal mapping guides; YOUR mapping makes pathways beg to be used.[/white]",
    ]))
    
    # Create debug logger if needed (write logs under the project's graphs dir)
    debug_logger = None
    if debug:
        try:
            debug_out = (Path(output_dir) / "graphs" / ".hound_debug").resolve()
        except Exception:
            debug_out = None
        debug_logger = DebugLogger(session_id=f"graph_{repo_name}_{int(time.time())}", output_dir=debug_out)
    
    try:
        files_to_include = [f.strip() for f in file_filter.split(",")] if file_filter else None
        if files_to_include and debug:
            console.print(f"[dim]File filter: {len(files_to_include)} specific files[/dim]")
        
        # Prepare unified live event log (consistent with agent UI)
        from rich.live import Live
        event_log: list[str] = []

        def _short(s: str, n: int = 120) -> str:
            return (s[: n - 3] + '...') if isinstance(s, str) and len(s) > n else (s or '')

        def _panel():
            content = "\n".join(event_log[-12:]) if event_log else "Initializing..."
            return Panel(content, title="[bold cyan]Graph Build Progress[/bold cyan]", border_style="cyan")

        use_live = (progress_console.is_terminal and not quiet)
        if use_live:
            live_ctx = Live(_panel(), console=progress_console, refresh_per_second=8, transient=True)
        else:
            # Dummy context manager
            from contextlib import contextmanager
            @contextmanager
            def live_ctx_manager():
                yield None
            live_ctx = live_ctx_manager()

        with live_ctx as live:
            def log_line(kind: str, msg: str):
                now = datetime.now().strftime('%H:%M:%S')
                colors = {
                    'ingest': 'bright_yellow', 'build': 'bright_cyan', 'discover': 'bright_magenta', 'graph': 'bright_cyan',
                    'sample': 'bright_white', 'update': 'bright_green', 'warn': 'bright_red', 'save': 'bright_green', 'phase': 'bright_blue',
                    'stats': 'bright_white', 'start': 'bright_white', 'complete': 'bright_green'
                }
                color = colors.get(kind, 'white')
                line = f"[{color}]{now}[/{color}] {msg}"
                event_log.append(line)
                if use_live and live is not None:
                    live.update(_panel())
                elif not quiet:
                    progress_console.print(line)

            # Step 1: Ingestion
            log_line('ingest', 'Step 1: Repository Ingestion')
            manifest = RepositoryManifest(str(repo_path), config, file_filter=files_to_include)
            cards, files = manifest.walk_repository()
            manifest.save_manifest(manifest_dir)
            log_line('ingest', f"Ingested {len(files)} files → {len(cards)} cards")

            bundler = AdaptiveBundler(cards, files, config)
            bundles = bundler.create_bundles()
            bundler.save_bundles(manifest_dir)
            log_line('ingest', f"Created {len(bundles)} bundles")

            # Step 2: Graph Building with detailed progress
            console.print("\n[bold]Step 2:[/bold] Graph Construction")
            focus_list = [f.strip() for f in focus_areas.split(",")] if focus_areas else None
            if focus_list:
                log_line('build', f"Focus areas: {', '.join(focus_list)}")

            builder = GraphBuilder(config, debug=debug, debug_logger=debug_logger)

            # Narrative model names: reflect effective models used by builder
            try:
                graph_model = getattr(builder.llm, 'model', None) or 'Graph-Model'
            except Exception:
                graph_model = 'Graph-Model'
            try:
                discovery_model = getattr(builder.llm_agent, 'model', None) or 'Guidance-Model'
            except Exception:
                discovery_model = 'Guidance-Model'

            # Animated progress bar during graph construction
            iteration_total = max_iterations
            if progress_console.is_terminal and not quiet:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=progress_console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(f"Constructing graphs (iteration 0/{iteration_total})...", total=iteration_total, completed=0)

                    def builder_callback(info):
                        # Handles dict payloads from GraphBuilder._emit
                        if isinstance(info, dict):
                            msg = info.get('message', '')
                            kind = info.get('status', 'build')
                            # Narrative seasoning + progress description
                            if kind == 'discover':
                                line = random.choice([
                                    f"🧑‍🏭 {discovery_model} scouts the terrain: {msg}",
                                    f"🧑‍🏭 Strategist {discovery_model} surveys the codebase — bold move!",
                                    msg,
                                ])
                                log_line(kind, line)
                                progress.update(task, description=_short(line, 80))
                            elif kind in ('graph_build', 'building'):
                                line = random.choice([
                                    f"🗺️  {graph_model} sketches connections — {msg}",
                                    msg,
                                ])
                                log_line(kind, line)
                                
                                # Parse iteration from building messages
                                import re
                                m = re.search(r"iteration\s+(\d+)/(\d+)", msg)
                                if m:
                                    cur = int(m.group(1))
                                    total = int(m.group(2))
                                    if total != progress.tasks[task].total:
                                        progress.update(task, total=total)
                                    # Update both completed and description
                                    completed = min(cur, total)
                                    progress.update(task, completed=completed, description=f"Constructing graphs (iteration {cur}/{total})...")
                                else:
                                    progress.update(task, description=_short(line, 80))
                            elif kind == 'update':
                                line = random.choice([
                                    f"🔧 {graph_model} chisels the graph: {msg}",
                                    msg,
                                ])
                                log_line(kind, line)
                                progress.update(task, description=_short(line, 80))
                            elif kind == 'save':
                                line = random.choice([
                                    f"💾 {graph_model} files the maps: {msg}",
                                    msg,
                                ])
                                log_line(kind, line)
                                progress.update(task, description=_short(line, 80))
                            else:
                                log_line(kind, msg)
                                # Try to parse iteration updates like "iteration X/Y"
                                import re
                                m = re.search(r"iteration\s+(\d+)/(\d+)", msg)
                                if m:
                                    cur = int(m.group(1))
                                    total = int(m.group(2))
                                    if total != progress.tasks[task].total:
                                        progress.update(task, total=total)
                                    # ensure non-decreasing
                                    completed = min(max(cur, progress.tasks[task].completed or 0), total)
                                    progress.update(task, completed=completed, description=f"Constructing graphs (iteration {cur}/{total})...")
                                else:
                                    # update description only
                                    progress.update(task, description=_short(msg, 80))
                        else:
                            text = str(info)
                            progress.update(task, description=_short(text, 80))
                            log_line('build', text)

                    results = builder.build(
                        manifest_dir=manifest_dir,
                        output_dir=graphs_dir,
                        max_iterations=max_iterations,
                        focus_areas=focus_list,
                        max_graphs=max_graphs,
                        progress_callback=builder_callback
                    )
            else:
                # Quiet/non-TTY: simple logging, no progress bar
                def builder_callback(info):
                    if isinstance(info, dict):
                        msg = info.get('message', '')
                        kind = info.get('status', 'build')
                        if not quiet:
                            if kind == 'discover':
                                log_line(kind, f"🧑‍🏭 {discovery_model} scouts the terrain: {msg}")
                            elif kind in ('graph_build','building'):
                                log_line(kind, f"🗺️  {graph_model} sketches connections — {msg}")
                            elif kind == 'update':
                                log_line(kind, f"🔧 {graph_model} chisels the graph: {msg}")
                            else:
                                log_line(kind, msg)
                    else:
                        if not quiet:
                            log_line('build', str(info))

                results = builder.build(
                    manifest_dir=manifest_dir,
                    output_dir=graphs_dir,
                    max_iterations=max_iterations,
                    focus_areas=focus_list,
                    max_graphs=max_graphs,
                    progress_callback=builder_callback
                )
    
            # Display results in a nice table
            if results['graphs']:
                table = Table(title="Generated Graphs", box=box.SIMPLE_HEAD)
                table.add_column("Graph", style="cyan", no_wrap=True)
                table.add_column("Nodes", justify="right", style="green")
                table.add_column("Edges", justify="right", style="green")
                table.add_column("Focus", style="dim")
        
                total_nodes = 0
                total_edges = 0
                
                for name, path in results['graphs'].items():
                    with open(path) as f:
                        graph_data = json.load(f)
                    stats = graph_data.get('stats', {})
                    nodes = stats.get('num_nodes', 0)
                    edges = stats.get('num_edges', 0)
                    focus = graph_data.get('focus', 'general')
                    
                    table.add_row(
                        graph_data.get('name', name),
                        str(nodes),
                        str(edges),
                        focus
                    )
                    total_nodes += nodes
                    total_edges += edges
                
                console.print(table)
                console.print(f"\n  [bold]Total:[/bold] {total_nodes} nodes, {total_edges} edges")
    
            # Step 3: Visualization
            if visualize:
                console.print("\n[bold]Step 3:[/bold] Visualization")
                if progress_console.is_terminal and not quiet:
                    # small spinner while generating viz
                    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=progress_console, transient=True) as p:
                        t = p.add_task("Generating interactive visualization...", total=None)
                        html_path = generate_dynamic_visualization(graphs_dir)
                        p.update(t, completed=1)
                else:
                    html_path = generate_dynamic_visualization(graphs_dir)
                log_line('save', f"Visualization saved: {html_path}")
                console.print(f"\n[bold]Open in browser:[/bold] [link]file://{html_path.resolve()}[/link]")
                console.print(f"\n[dim]Tip: Use 'hound graph export {repo_name} --open' to regenerate and open visualization[/dim]")
    
        # Finalize debug log if enabled
        if debug and debug_logger:
            log_path = debug_logger.finalize()
            console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")
        
        console.print(Panel.fit(
            "[green]✓[/green] Graph building complete!",
            box=box.ROUNDED,
            style="green"
        ))
    
    except Exception as e:
        console.print(f"[red]Error during graph building: {e}[/red]")
        # Finalize debug log on error
        if debug and debug_logger:
            log_path = debug_logger.finalize()
            console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")
        raise


def ingest(
    repo_path: str = typer.Argument(..., help="Path to repository to analyze"),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    file_filter: str | None = typer.Option(None, "--files", "-f", help="Comma-separated file paths"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
):
    """Ingest repository and create bundles."""
    from ingest.bundles import AdaptiveBundler
    from ingest.manifest import RepositoryManifest
    
    config = load_config(config_path)
    
    files_to_include = [f.strip() for f in file_filter.split(",")] if file_filter else None
    output_dir = Path(output_dir) if output_dir else Path(".hound_cache") / Path(repo_path).name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Header
    console.print(Panel.fit(
        f"[bold cyan]Repository Ingestion[/bold cyan]\n"
        f"Path: [white]{repo_path}[/white]\n"
        f"Output: [white]{output_dir}[/white]",
        box=box.ROUNDED
    ))
    
    if files_to_include and debug:
        console.print(f"[dim]File filter: {len(files_to_include)} specific files[/dim]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        # Create manifest
        task1 = progress.add_task("Creating repository manifest...", total=100)
        manifest = RepositoryManifest(repo_path, config, file_filter=files_to_include)
        cards, files = manifest.walk_repository()
        manifest_info = manifest.save_manifest(output_dir)
        progress.update(task1, completed=100)
        
        # Create bundles
        task2 = progress.add_task("Creating adaptive bundles...", total=100)
        bundler = AdaptiveBundler(cards, files, config)
        bundles = bundler.create_bundles()
        bundle_summary = bundler.save_bundles(output_dir)
        progress.update(task2, completed=100)
    
    # Results summary
    console.print(f"\n[green]✓[/green] Created [bold]{len(cards)}[/bold] cards from [bold]{len(files)}[/bold] files")
    console.print(f"  Total size: [cyan]{manifest_info['total_chars']:,}[/cyan] characters")
    console.print(f"\n[green]✓[/green] Created [bold]{len(bundles)}[/bold] bundles")
    console.print(f"  Average size: [cyan]{bundle_summary['avg_bundle_size']:,.0f}[/cyan] chars")
    console.print(f"  Range: [cyan]{bundle_summary['min_bundle_size']:,}[/cyan] - [cyan]{bundle_summary['max_bundle_size']:,}[/cyan] chars")
    
    # Bundle details table (only in debug mode)
    if bundles and debug:
        table = Table(title="Bundle Summary", box=box.SIMPLE_HEAD)
        table.add_column("Bundle ID", style="cyan")
        table.add_column("Cards", justify="right")
        table.add_column("Files", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Preview", style="dim")
        
        for bundle in bundles[:10]:
            table.add_row(
                bundle.id,
                str(len(bundle.card_ids)),
                str(len(bundle.file_paths)),
                f"{bundle.total_chars:,}",
                bundle.preview[:50] + "..." if len(bundle.preview) > 50 else bundle.preview
            )
        
        if len(bundles) > 10:
            table.add_row("...", "...", "...", "...", f"({len(bundles) - 10} more bundles)")
        
        console.print(table)
    
    console.print(Panel.fit(
        f"[green]✓[/green] Ingestion complete!\nOutput saved to: [cyan]{output_dir}[/cyan]",
        box=box.ROUNDED,
        style="green"
    ))
