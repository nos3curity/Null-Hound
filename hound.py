#!/usr/bin/env python3
"""Hound - AI-powered security analysis system."""

import sys
import os
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

from commands.graph import build, ingest
from commands.project import ProjectManager

app = typer.Typer(
    name="hound",
    help="Cracked security analysis agents",
    add_completion=False,
)
console = Console()

# Create project subcommand group
project_app = typer.Typer(help="Manage Hound projects")
app.add_typer(project_app, name="project")

# Create agent subcommand group
agent_app = typer.Typer(help="Run security analysis agents")
app.add_typer(agent_app, name="agent")

@project_app.command("create")
def project_create(
    name: str = typer.Argument(..., help="Project name"),
    source_path: str = typer.Argument(..., help="Path to source code"),
    description: str = typer.Option(None, "--description", "-d", help="Project description"),
    auto_name: bool = typer.Option(False, "--auto-name", "-a", help="Auto-generate project name")
):
    """Create a new project."""
    from commands.project import create
    import click
    ctx = click.Context(create)
    ctx.params = {
        'name': name,
        'source_path': source_path,
        'description': description,
        'auto_name': auto_name
    }
    create.invoke(ctx)

@project_app.command("list")
def project_list():
    """List all projects."""
    from commands.project import list_projects_cmd
    import click
    ctx = click.Context(list_projects_cmd)
    ctx.params = {'output_json': False}
    list_projects_cmd.invoke(ctx)

@project_app.command("info")
def project_info(name: str = typer.Argument(..., help="Project name")):
    """Show project information."""
    from commands.project import info
    import click
    ctx = click.Context(info)
    ctx.params = {'name': name}
    info.invoke(ctx)

@project_app.command("delete")
def project_delete(
    name: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force delete without confirmation")
):
    """Delete a project."""
    from commands.project import delete
    import click
    ctx = click.Context(delete)
    ctx.params = {'name': name, 'force': force}
    delete.invoke(ctx)

@project_app.command("hypotheses")
def project_hypotheses(
    name: str = typer.Argument(..., help="Project name"),
    details: bool = typer.Option(False, "--details", "-d", help="Show full descriptions without abbreviation")
):
    """List all hypotheses for a project with confidence ratings."""
    from commands.project import hypotheses
    import click
    ctx = click.Context(hypotheses)
    ctx.params = {'name': name, 'details': details}
    hypotheses.invoke(ctx)

# Agent audit subcommand
@agent_app.command("audit")
def agent_audit(
    target: str = typer.Argument(None, help="Project name or path (optional with --project)"),
    iterations: int = typer.Option(20, "--iterations", help="Maximum iterations per investigation (default: 20)"),
    plan_n: int = typer.Option(5, "--plan-n", help="Number of investigations to plan per batch (default: 5)"),
    time_limit: int = typer.Option(None, "--time-limit", help="Time limit in minutes"),
    config: str = typer.Option(None, "--config", help="Configuration file"),
    resume: bool = typer.Option(False, "--resume", help="Resume from previous session"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    platform: str = typer.Option(None, "--platform", help="Override LLM platform (e.g., openai, anthropic)"),
    model: str = typer.Option(None, "--model", help="Override LLM model (e.g., gpt-4, claude-3)")
):
    """Run autonomous security audit (plans investigations automatically)."""
    from commands.agent import agent as agent_command
    import click
    
    manager = ProjectManager()
    project_id = None
    
    # Handle different input modes
    if project:
        # Use specified project
        proj = manager.get_project(project)
        if not proj:
            console.print(f"[red]Project '{project}' not found.[/red]")
            raise typer.Exit(1)
        project_id = str(manager.get_project_path(project))
        console.print(f"[cyan]Using project:[/cyan] {project}")
    elif target:
        # Check if target is a project name or path
        proj = manager.get_project(target)
        if proj:
            project_id = str(manager.get_project_path(target))
            console.print(f"[cyan]Using project:[/cyan] {target}")
        else:
            # Target is a path
            project_id = target
    else:
        # No target or project specified
        console.print("[red]Error: Either specify a target path/project name or use --project option[/red]")
        raise typer.Exit(1)
    
    console.print("[bold cyan]Running autonomous audit...[/bold cyan]")
    
    # Create a Click context and invoke the command
    ctx = click.Context(agent_command)
    ctx.params = {
        'project_id': project_id,
        'iterations': iterations,
        'plan_n': plan_n,
        'time_limit': time_limit,
        'config': config,
        'resume': resume,
        'debug': debug,
        'platform': platform,
        'model': model
    }
    agent_command.invoke(ctx)


# Agent investigate subcommand
@agent_app.command("investigate")
def agent_investigate(
    prompt: str = typer.Argument(..., help="Investigation prompt or question"),
    target: str = typer.Argument(None, help="Project name or path (optional with --project)"),
    iterations: int = typer.Option(None, "--iterations", help="Maximum iterations for the agent"),
    config: str = typer.Option(None, "--config", help="Configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    platform: str = typer.Option(None, "--platform", help="Override LLM platform (e.g., openai, anthropic)"),
    model: str = typer.Option(None, "--model", help="Override LLM model (e.g., gpt-4, claude-3)")
):
    """Run targeted investigation with a specific prompt."""
    manager = ProjectManager()
    project_id = None
    
    # Handle different input modes
    if project:
        # Use specified project
        proj = manager.get_project(project)
        if not proj:
            console.print(f"[red]Project '{project}' not found.[/red]")
            raise typer.Exit(1)
        project_id = str(manager.get_project_path(project))
        console.print(f"[cyan]Using project:[/cyan] {project}")
    elif target:
        # Check if target is a project name or path
        proj = manager.get_project(target)
        if proj:
            project_id = str(manager.get_project_path(target))
            console.print(f"[cyan]Using project:[/cyan] {target}")
        else:
            # Target is a path
            project_id = target
    else:
        # No target or project specified
        console.print("[red]Error: Either specify a target path/project name or use --project option[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]Investigation:[/bold cyan] {prompt}")
    
    # Run the investigation using the agent's investigate method
    from commands.agent import run_investigation
    run_investigation(
        project_path=project_id,
        prompt=prompt,
        iterations=iterations,
        config_path=Path(config) if config else None,
        debug=debug,
        platform=platform,
        model=model
    )


# Create graph subcommand group
graph_app = typer.Typer(help="Build and manage knowledge graphs")
app.add_typer(graph_app, name="graph")

@graph_app.command("build")
def graph_build(
    target: str = typer.Argument(None, help="Project name or source path (optional with --project)"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    create_project: bool = typer.Option(False, "--create-project", help="Create a new project"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Maximum iterations for graph refinement"),
    graphs: int = typer.Option(2, "--graphs", "-g", help="Number of graphs to generate"),
    focus: str = typer.Option(None, "--focus", "-f", help="Comma-separated focus areas"),
    files: str = typer.Option(None, "--files", help="Comma-separated list of file paths to include"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output and disable animations"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output")
):
    """Build system architecture graph from source code."""
    from commands.graph import build
    import click
    
    manager = ProjectManager()
    source_path = None
    output_dir = output
    
    # Handle different input modes
    if project:
        # Use specified project
        proj = manager.get_project(project)
        if not proj:
            console.print(f"[red]Project '{project}' not found.[/red]")
            raise typer.Exit(1)
        project_path = manager.get_project_path(project)
        source_path = proj['source_path']
        output_dir = str(project_path)  # Don't add /graphs here, graph.py will add it
        console.print(f"[cyan]Using project:[/cyan] {project}")
    elif target:
        # Check if target is a project name or path
        proj = manager.get_project(target)
        if proj:
            # Target is a project name
            project_path = manager.get_project_path(target)
            source_path = proj['source_path']
            output_dir = str(project_path)  # Don't add /graphs here, graph.py will add it
            console.print(f"[cyan]Using project:[/cyan] {target}")
        else:
            # Target is a source path
            source_path = target
            if create_project:
                # Create new project from this path
                project_name = Path(target).name
                proj_config = manager.create_project(
                    project_name, target, 
                    f"Graph analysis of {project_name}",
                    auto_name=True
                )
                project_path = manager.get_project_path(proj_config['name'])
                output_dir = str(project_path)  # Don't add /graphs here, graph.py will add it
                console.print(f"[green]Created project:[/green] {proj_config['name']}")
    else:
        # No target or project specified
        console.print("[red]Error: Either specify a target path/project name or use --project option[/red]")
        raise typer.Exit(1)
    
    # Run graph build directly
    build(
        repo_path=source_path,
        repo_id=project if project else target,
        output_dir=output_dir,
        config_path=Path(config) if config else None,
        max_iterations=iterations,
        max_graphs=graphs,
        focus_areas=focus,
        file_filter=files,
        visualize=True,
        debug=debug,
        quiet=quiet
    )

@graph_app.command("add-custom")
def graph_add_custom(
    target: str = typer.Argument(..., help="Project name"),
    description: str = typer.Argument(..., help="Graph description (e.g., 'authentication roles vs components')"),
    name: str = typer.Option(None, "--name", "-n", help="Custom name for the graph"),
    iterations: int = typer.Option(1, "--iterations", "-i", help="Number of refinement iterations"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project (alternative to target)")
):
    """Add a custom graph with user-defined focus."""
    import json
    
    manager = ProjectManager()
    project_name = project or target
    
    if not project_name:
        console.print("[red]Error: Specify a project name or use --project option[/red]")
        raise typer.Exit(1)
    
    # Get project
    proj = manager.get_project(project_name)
    if not proj:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        raise typer.Exit(1)
    
    project_path = manager.get_project_path(project_name)
    graphs_dir = project_path / "graphs"
    
    # Check if system graph exists (try both common names)
    system_graph_path = graphs_dir / "graph_SystemArchitecture.json"
    if not system_graph_path.exists():
        # Try alternative name
        system_graph_path = graphs_dir / "graph_SystemOverview.json"
        if not system_graph_path.exists():
            console.print("[red]Error: System architecture graph not found.[/red]")
            # Get the actual command used to run this script
            cli_cmd = os.path.basename(sys.argv[0]) if sys.argv else "hound"
            if cli_cmd.endswith('.py'):
                if sys.argv and sys.argv[0].startswith('./'):
                    cli_cmd = sys.argv[0]
                else:
                    cli_cmd = f"python {cli_cmd}"
            console.print(f"[yellow]Run '{cli_cmd} graph build' first to create the base graph.[/yellow]")
            raise typer.Exit(1)
    
    # Generate graph name
    if not name:
        # Pass None to let the LLM generate a meaningful name
        name = None
    
    # Create the custom graph using the graph builder
    console.print(f"[cyan]Creating custom graph:[/cyan] {description}")
    if iterations > 1:
        console.print(f"  [dim]Using {iterations} refinement iterations[/dim]")
    
    # Use the custom graph builder that reuses main logic
    from commands.graph_custom import build_custom_graph
    
    try:
        # Build the custom graph
        config_path = Path(config) if config else None
        custom_graph_path = build_custom_graph(
            project_path=project_path,
            description=description,
            name=name,
            config_path=config_path,
            iterations=iterations,
            debug=False  # Could add --debug flag if needed
        )
        
        console.print(f"[green]✓ Custom graph created:[/green] {custom_graph_path}")
        
        # Load and show summary
        with open(custom_graph_path, 'r') as f:
            graph_data = json.load(f)
        
        stats = graph_data.get('stats', {})
        console.print(f"  Nodes: {stats.get('num_nodes', len(graph_data.get('nodes', [])))}")
        console.print(f"  Edges: {stats.get('num_edges', len(graph_data.get('edges', [])))}")
        if stats.get('iterations'):
            console.print(f"  Iterations: {stats['iterations']}")
        
        console.print(f"\n[cyan]To analyze with this graph, use:[/cyan]")
        # Get the actual command used to run this script
        cli_cmd = os.path.basename(sys.argv[0]) if sys.argv else "hound"
        if cli_cmd.endswith('.py'):
            if sys.argv and sys.argv[0].startswith('./'):
                cli_cmd = sys.argv[0]
            else:
                cli_cmd = f"python {cli_cmd}"
        console.print(f"  {cli_cmd} agent audit --project {project_name}")
        console.print(f"  {cli_cmd} agent investigate \"<your question>\" --project {project_name}")
        
    except Exception as e:
        console.print(f"[red]Error creating custom graph:[/red] {str(e)}")
        raise typer.Exit(1)


@graph_app.command("export")
def graph_export(
    target: str = typer.Argument(..., help="Project name"),
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file path"),
    open_browser: bool = typer.Option(False, "--open", help="Open visualization in browser"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project (alternative to target)")
):
    """Export graphs to interactive HTML visualization."""
    from visualization.dynamic_graph_viz import generate_dynamic_visualization
    
    manager = ProjectManager()
    project_name = project or target
    
    if not project_name:
        console.print("[red]Error: Specify a project name or use --project option[/red]")
        raise typer.Exit(1)
    
    # Get project
    proj = manager.get_project(project_name)
    if not proj:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        raise typer.Exit(1)
    
    project_path = manager.get_project_path(project_name)
    graphs_dir = project_path / "graphs"
    
    # Check if graphs exist
    graph_files = list(graphs_dir.glob("graph_*.json"))
    if not graph_files:
        console.print("[red]Error: No graphs found in project.[/red]")
        console.print("[yellow]Run 'graph build' or 'graph add-custom' first to create graphs.[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Exporting graphs for project:[/cyan] {project_name}")
    console.print(f"  Found {len(graph_files)} graphs")
    
    # Generate visualization
    try:
        output_path = Path(output) if output else None
        html_path = generate_dynamic_visualization(graphs_dir, output_path)
        
        console.print(f"\n[green]✓ Visualization exported to:[/green] {html_path}")
        
        # Open in browser if requested
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{html_path.resolve()}")
            console.print(f"[green]✓ Opened in browser[/green]")
        else:
            console.print(f"\n[bold]Open in browser:[/bold]")
            console.print(f"  [link]file://{html_path.resolve()}[/link]")
            
            # If on macOS, offer to open in browser
            import platform
            if platform.system() == "Darwin":
                console.print(f"\n[dim]Or run: open {html_path}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error exporting visualization:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def finalize(
    project: str = typer.Argument(..., help="Project name"),
    threshold: float = typer.Option(0.75, "--threshold", "-t", help="Confidence threshold for review"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Finalize hypotheses - review and confirm/reject high-confidence findings."""
    from commands.finalize import finalize as finalize_command
    import click
    
    console.print("[bold cyan]Running hypothesis finalization...[/bold cyan]")
    
    # Create Click context and invoke
    ctx = click.Context(finalize_command)
    ctx.params = {
        'project_name': project,
        'threshold': threshold,
        'skip_filter': False,  # Default to not skipping the filter
        'debug': debug
    }
    
    try:
        finalize_command.invoke(ctx)
    except SystemExit as e:
        if e.code != 0:
            raise typer.Exit(e.code)


@app.command()
def report(
    project: str = typer.Argument(..., help="Project name"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("html", "--format", "-f", help="Report format (html/markdown)"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom report title"),
    auditors: str = typer.Option("Security Team", "--auditors", "-a", help="Comma-separated auditor names"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Generate a professional security audit report."""
    from commands.report import report as report_command
    import click
    
    console.print("[bold cyan]Generating security audit report...[/bold cyan]")
    
    # Create Click context and invoke
    ctx = click.Context(report_command)
    ctx.params = {
        'project_name': project,
        'output': output,
        'format': format,
        'title': title,
        'auditors': auditors,
        'debug': debug,
        'show_prompt': False  # Add missing parameter
    }
    
    try:
        report_command.invoke(ctx)
    except click.Exit as e:
        raise typer.Exit(e.exit_code)


@app.command()
def version():
    """Show Hound version."""
    console.print("[bold]Hound[/bold] v2.0.0")
    console.print("AI-powered security analysis system")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
