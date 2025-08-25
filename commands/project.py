"""
Project management commands for Hound.

Projects organize analysis results and configurations for specific codebases.
"""

import json
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


class ProjectManager:
    """Manages Hound projects."""
    
    def __init__(self):
        self.projects_dir = Path.home() / ".hound" / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.projects_dir / "registry.json"
        self._ensure_registry()
    
    def _ensure_registry(self):
        """Ensure project registry exists."""
        if not self.registry_file.exists():
            with open(str(self.registry_file), 'w') as f:
                json.dump({"projects": {}}, f, indent=2)
    
    def _load_registry(self) -> Dict:
        """Load project registry."""
        with open(str(self.registry_file), 'r') as f:
            return json.load(f)
    
    def _save_registry(self, registry: Dict):
        """Save project registry."""
        with open(str(self.registry_file), 'w') as f:
            json.dump(registry, f, indent=2)
    
    def create_project(self, name: str, source_path: str, 
                      description: Optional[str] = None,
                      auto_name: bool = False) -> Dict:
        """Create a new project."""
        source_path = Path(source_path).resolve()
        
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Auto-generate name if requested
        if auto_name:
            name = f"{source_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if project already exists
        registry = self._load_registry()
        if name in registry["projects"]:
            raise ValueError(f"Project '{name}' already exists")
        
        # Create project directory
        project_dir = self.projects_dir / name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project subdirectories
        (project_dir / "graphs").mkdir(exist_ok=True)
        (project_dir / "manifest").mkdir(exist_ok=True)
        (project_dir / "agent_runs").mkdir(exist_ok=True)
        (project_dir / "reports").mkdir(exist_ok=True)
        
        # Create project config
        project_config = {
            "name": name,
            "source_path": str(source_path),
            "description": description or f"Analysis of {source_path.name}",
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Save project config
        with open(project_dir / "project.json", 'w') as f:
            json.dump(project_config, f, indent=2)
        
        # Update registry
        registry["projects"][name] = {
            "path": str(project_dir),
            "source_path": str(source_path),
            "created_at": project_config["created_at"],
            "description": project_config["description"]
        }
        self._save_registry(registry)
        
        return project_config
    
    def list_projects(self) -> List[Dict]:
        """List all projects."""
        registry = self._load_registry()
        projects = []
        
        for name, info in registry["projects"].items():
            project_dir = Path(info["path"])
            if project_dir.exists():
                # Load full project config
                config_file = project_dir / "project.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Check for analysis results
                    graphs_exist = len(list((project_dir / "graphs").glob("*.json"))) > 0
                    runs_count = len(list((project_dir / "agent_runs").glob("*.json")))
                    
                    projects.append({
                        "name": name,
                        "source_path": config["source_path"],
                        "description": config["description"],
                        "created_at": config["created_at"],
                        "last_accessed": config.get("last_accessed", ""),
                        "has_graphs": graphs_exist,
                        "agent_runs": runs_count,
                        "path": str(project_dir)
                    })
        
        return projects
    
    def get_project(self, name: str) -> Optional[Dict]:
        """Get project by name."""
        registry = self._load_registry()
        
        if name not in registry["projects"]:
            return None
        
        project_dir = Path(registry["projects"][name]["path"])
        if not project_dir.exists():
            return None
        
        config_file = project_dir / "project.json"
        if not config_file.exists():
            return None
        
        import time
        import fcntl
        
        # Retry logic for reading JSON with file locking
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with open(config_file, 'r') as f:
                    # Try to get a shared lock for reading
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                        content = f.read()
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except IOError:
                        # If we can't get lock, just read anyway
                        content = f.read()
                    
                    if not content:
                        # Empty file, retry
                        if attempt < max_retries - 1:
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        else:
                            return None
                    
                    config = json.loads(content)
                    break
            except (json.JSONDecodeError, IOError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    # After all retries, return None
                    return None
        
        # Update last accessed with file locking
        try:
            config["last_accessed"] = datetime.now().isoformat()
            with open(config_file, 'w') as f:
                # Try to get exclusive lock for writing
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    json.dump(config, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except IOError:
                    # If we can't get lock, skip updating last_accessed
                    pass
        except Exception:
            # If update fails, continue anyway - it's not critical
            pass
        
        config["path"] = str(project_dir)
        return config
    
    def delete_project(self, name: str, force: bool = False) -> bool:
        """Delete a project."""
        registry = self._load_registry()
        
        if name not in registry["projects"]:
            return False
        
        project_dir = Path(registry["projects"][name]["path"])
        
        if not force:
            # Check if project has important data
            has_data = False
            if project_dir.exists():
                graphs_count = len(list((project_dir / "graphs").glob("*.json")))
                runs_count = len(list((project_dir / "agent_runs").glob("*.json")))
                has_data = graphs_count > 0 or runs_count > 0
            
            if has_data and not Confirm.ask(
                f"[yellow]Project '{name}' contains {graphs_count} graphs and {runs_count} agent runs. "
                "Delete anyway?[/yellow]"
            ):
                return False
        
        # Remove project directory
        if project_dir.exists():
            shutil.rmtree(project_dir)
        
        # Update registry
        del registry["projects"][name]
        self._save_registry(registry)
        
        return True
    
    def get_project_path(self, name: str) -> Optional[Path]:
        """Get project directory path."""
        project = self.get_project(name)
        if project:
            return Path(project["path"])
        return None


# CLI Commands

@click.group()
def project():
    """Manage Hound projects."""
    pass


@project.command()
@click.argument('name')
@click.argument('source_path')
@click.option('--description', '-d', help="Project description")
@click.option('--auto-name', '-a', is_flag=True, help="Auto-generate project name")
def create(name: str, source_path: str, description: Optional[str], auto_name: bool):
    """Create a new project."""
    manager = ProjectManager()
    
    try:
        config = manager.create_project(name, source_path, description, auto_name)
        
        console.print(Panel(
            f"[green]✓ Project created successfully![/green]\n\n"
            f"[bold]Name:[/bold] {config['name']}\n"
            f"[bold]Source:[/bold] {config['source_path']}\n"
            f"[bold]Description:[/bold] {config['description']}\n\n"
            f"[dim]Project directory: {manager.projects_dir / config['name']}[/dim]",
            title="[bold cyan]New Project[/bold cyan]",
            border_style="green"
        ))
        
        # Get the actual command used to run this script
        cli_cmd = os.path.basename(sys.argv[0]) if sys.argv else "hound"
        # If it's a python script, include python/python3
        if cli_cmd.endswith('.py'):
            # Check if it was run directly (./script.py) or via python
            if sys.argv and sys.argv[0].startswith('./'):
                cli_cmd = sys.argv[0]
            else:
                cli_cmd = f"python {cli_cmd}"
        
        console.print(f"\n[cyan]To analyze this project, run:[/cyan]")
        console.print(f"  {cli_cmd} graph build --project {config['name']}")
        console.print(f"  {cli_cmd} agent audit --project {config['name']}")
        
    except ValueError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise click.Exit(1)


@project.command(name='list')
@click.option('--json', 'output_json', is_flag=True, help="Output as JSON")
def list_projects_cmd(output_json: bool):
    """List all projects."""
    manager = ProjectManager()
    projects = manager.list_projects()
    
    if output_json:
        click.echo(json.dumps(projects, indent=2))
        return
    
    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        console.print("\n[cyan]Create a project with:[/cyan]")
        # Get the actual command used to run this script
        cli_cmd = os.path.basename(sys.argv[0]) if sys.argv else "hound"
        if cli_cmd.endswith('.py'):
            if sys.argv and sys.argv[0].startswith('./'):
                cli_cmd = sys.argv[0]
            else:
                cli_cmd = f"python {cli_cmd}"
        console.print(f"  {cli_cmd} project create <name> <source_path>")
        return
    
    # Create table
    table = Table(title="[bold cyan]Hound Projects[/bold cyan]")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="white")
    table.add_column("Description", style="dim")
    table.add_column("Graphs", style="green")
    table.add_column("Runs", style="yellow")
    table.add_column("Created", style="dim")
    
    for proj in sorted(projects, key=lambda x: x["created_at"], reverse=True):
        source = Path(proj["source_path"])
        source_display = f".../{source.parent.name}/{source.name}" if len(str(source)) > 40 else str(source)
        
        table.add_row(
            proj["name"],
            source_display,
            proj["description"][:30] + "..." if len(proj["description"]) > 30 else proj["description"],
            "✓" if proj["has_graphs"] else "-",
            str(proj["agent_runs"]) if proj["agent_runs"] > 0 else "-",
            proj["created_at"].split("T")[0]
        )
    
    console.print(table)
    console.print(f"\n[dim]Total projects: {len(projects)}[/dim]")


@project.command()
@click.argument('name')
def info(name: str):
    """Show detailed project information."""
    manager = ProjectManager()
    project = manager.get_project(name)
    
    if not project:
        console.print(f"[red]Project '{name}' not found.[/red]")
        raise click.Exit(1)
    
    project_dir = Path(project["path"])
    
    # Gather statistics
    graphs_files = list((project_dir / "graphs").glob("*.json"))
    manifest_files = list((project_dir / "manifest").glob("*"))
    agent_runs = list((project_dir / "agent_runs").glob("*.json"))
    reports = list((project_dir / "reports").glob("*"))
    
    # Check for hypotheses
    hypothesis_stats = {"total": 0, "confirmed": 0, "high_confidence": 0}
    hypothesis_file = project_dir / "hypotheses.json"
    if hypothesis_file.exists():
        try:
            with open(hypothesis_file, 'r') as f:
                hyp_data = json.load(f)
                hypotheses = hyp_data.get("hypotheses", {})
                hypothesis_stats["total"] = len(hypotheses)
                hypothesis_stats["confirmed"] = sum(1 for h in hypotheses.values() if h.get("status") == "confirmed")
                hypothesis_stats["high_confidence"] = sum(1 for h in hypotheses.values() if h.get("confidence", 0) >= 0.75)
        except Exception:
            pass
    
    # Display info
    console.print(Panel(
        f"[bold cyan]{project['name']}[/bold cyan]\n\n"
        f"[bold]Source:[/bold] {project['source_path']}\n"
        f"[bold]Description:[/bold] {project['description']}\n"
        f"[bold]Created:[/bold] {project['created_at']}\n"
        f"[bold]Last accessed:[/bold] {project.get('last_accessed', 'Never')}\n\n"
        f"[bold]Analysis Results:[/bold]\n"
        f"  • Graphs: {len(graphs_files)} files\n"
        f"  • Manifest: {len(manifest_files)} files\n"
        f"  • Agent runs: {len(agent_runs)}\n"
        f"  • Reports: {len(reports)}\n"
        f"  • Hypotheses: {hypothesis_stats['total']} total"
        f" ([green]{hypothesis_stats['confirmed']} confirmed[/green],"
        f" [yellow]{hypothesis_stats['high_confidence']} high-confidence[/yellow])\n\n"
        f"[dim]Project directory: {project_dir}[/dim]",
        title="[bold]Project Information[/bold]",
        border_style="cyan"
    ))
    
    if graphs_files:
        console.print("\n[bold]Recent graphs:[/bold]")
        for graph_file in sorted(graphs_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            console.print(f"  • {graph_file.name}")
    
    if agent_runs:
        console.print("\n[bold]Recent agent runs:[/bold]")
        for run_file in sorted(agent_runs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            console.print(f"  • {run_file.name}")
    
    # Show top hypotheses if any exist
    if hypothesis_file.exists() and hypothesis_stats["total"] > 0:
        console.print("\n[bold]Top Hypotheses (by confidence):[/bold]")
        try:
            with open(hypothesis_file, 'r') as f:
                hyp_data = json.load(f)
                hypotheses = hyp_data.get("hypotheses", {})
                
                # Sort by confidence and get top 5
                sorted_hyps = sorted(
                    hypotheses.items(), 
                    key=lambda x: x[1].get("confidence", 0), 
                    reverse=True
                )[:5]
                
                for hyp_id, hyp in sorted_hyps:
                    conf = hyp.get("confidence", 0)
                    status = hyp.get("status", "proposed")
                    title = hyp.get("title", "Unknown")
                    vuln_type = hyp.get("vulnerability_type", "unknown")
                    
                    # Color code by confidence
                    if conf >= 0.8:
                        conf_color = "green"
                    elif conf >= 0.5:
                        conf_color = "yellow"
                    else:
                        conf_color = "red"
                    
                    # Status icon
                    if status == "confirmed":
                        status_icon = "✓"
                    elif status == "rejected":
                        status_icon = "✗"
                    else:
                        status_icon = "?"
                    
                    console.print(
                        f"  {status_icon} [{conf_color}]{conf:.0%}[/{conf_color}] "
                        f"{title[:60]} [dim]({vuln_type})[/dim]"
                    )
        except Exception:
            pass


@project.command()
@click.argument('name')
@click.option('--force', '-f', is_flag=True, help="Force delete without confirmation")
def delete(name: str, force: bool):
    """Delete a project."""
    manager = ProjectManager()
    
    if manager.delete_project(name, force):
        console.print(f"[green]✓ Project '{name}' deleted successfully.[/green]")
    else:
        console.print(f"[red]Failed to delete project '{name}'.[/red]")
        raise click.Exit(1)


@project.command()
@click.argument('name')
@click.option('--details', '-d', is_flag=True, help='Show full descriptions without abbreviation')
def hypotheses(name: str, details: bool = False):
    """List all hypotheses for a project with confidence ratings."""
    manager = ProjectManager()
    project = manager.get_project(name)
    
    if not project:
        console.print(f"[red]Project '{name}' not found.[/red]")
        raise click.Exit(1)
    
    project_dir = Path(project["path"])
    hypothesis_file = project_dir / "hypotheses.json"
    
    if not hypothesis_file.exists():
        console.print("[yellow]No hypotheses found for this project.[/yellow]")
        console.print("Run an investigation first with: hound agent investigate")
        raise click.Exit(0)
    
    # Load hypotheses
    with open(hypothesis_file, 'r') as f:
        hyp_data = json.load(f)
    
    hypotheses = hyp_data.get("hypotheses", {})
    
    if not hypotheses:
        console.print("[yellow]No hypotheses recorded yet.[/yellow]")
        raise click.Exit(0)
    
    # Create table
    from rich.table import Table
    
    table = Table(show_header=True, header_style="bold cyan", title=f"Hypotheses for {name}")
    
    if details:
        # Detailed view with full descriptions
        table.add_column("ID", style="dim", width=16)
        table.add_column("Title", overflow="fold")  # No width limit, allow full wrapping
        table.add_column("Description", overflow="fold")  # Add description column
        table.add_column("Type", style="cyan", width=18)
        table.add_column("Model", style="dim", overflow="fold")  # Allow model to wrap
        table.add_column("Confidence", justify="center", width=10)
        table.add_column("Status", justify="center", width=14)
        table.add_column("Severity", justify="center", width=10)
    else:
        # Compact view
        table.add_column("ID", style="dim", width=16)
        table.add_column("Title", width=50, overflow="fold")  # Allow full title with wrapping
        table.add_column("Type", style="cyan", width=18)
        table.add_column("Model", style="dim", width=20, overflow="ellipsis")  # Add model column
        table.add_column("Confidence", justify="center", width=10)
        table.add_column("Status", justify="center", width=14)
        table.add_column("Severity", justify="center", width=10)
    
    # Sort by confidence (highest first)
    sorted_hyps = sorted(
        hypotheses.items(),
        key=lambda x: x[1].get("confidence", 0),
        reverse=True
    )
    
    for hyp_id, hyp in sorted_hyps:
        # Format confidence with color
        conf = hyp.get("confidence", 0)
        if conf >= 0.8:
            conf_str = f"[bold green]{conf:.0%}[/bold green]"
        elif conf >= 0.5:
            conf_str = f"[yellow]{conf:.0%}[/yellow]"
        else:
            conf_str = f"[red]{conf:.0%}[/red]"
        
        # Format status with color
        status = hyp.get("status", "proposed")
        if status == "confirmed":
            status_str = "[bold green]✓ confirmed[/bold green]"
        elif status == "rejected":
            status_str = "[red]✗ rejected[/red]"
        elif status == "investigating":
            status_str = "[yellow]? investigating[/yellow]"
        elif status == "supported":
            status_str = "[cyan]+ supported[/cyan]"
        elif status == "refuted":
            status_str = "[magenta]- refuted[/magenta]"
        else:
            status_str = "[dim]○ proposed[/dim]"
        
        # Format severity
        severity = hyp.get("severity", "unknown")
        if severity == "critical":
            sev_str = "[bold red]CRITICAL[/bold red]"
        elif severity == "high":
            sev_str = "[red]HIGH[/red]"
        elif severity == "medium":
            sev_str = "[yellow]MEDIUM[/yellow]"
        elif severity == "low":
            sev_str = "[green]LOW[/green]"
        else:
            sev_str = "[dim]unknown[/dim]"
        
        # Get model info - show both junior and senior if available
        junior = hyp.get("junior_model")
        senior = hyp.get("senior_model")
        
        if junior and senior:
            model = f"J:{junior.split(':')[-1]} S:{senior.split(':')[-1]}"
        elif junior:
            model = f"J:{junior.split(':')[-1]}"
        elif senior:
            model = f"S:{senior.split(':')[-1]}"
        else:
            # Fallback to legacy field
            model = hyp.get("reported_by_model", "unknown")
            if ':' in model:
                model = model.split(':')[-1]
        
        if details:
            # Include full description in detailed view
            description = hyp.get("description", "No description available")
            table.add_row(
                hyp_id[:16],
                hyp.get("title", "Unknown"),
                description,  # Full description
                hyp.get("vulnerability_type", "unknown"),
                model,
                conf_str,
                status_str,
                sev_str
            )
        else:
            # Compact view without description
            table.add_row(
                hyp_id[:16],
                hyp.get("title", "Unknown"),  # Show full title, let Rich handle wrapping
                hyp.get("vulnerability_type", "unknown"),
                model,  # Add model column
                conf_str,
                status_str,
                sev_str
            )
    
    console.print(table)
    
    # Summary stats
    metadata = hyp_data.get("metadata", {})
    total = len(hypotheses)
    confirmed = sum(1 for h in hypotheses.values() if h.get("status") == "confirmed")
    high_conf = sum(1 for h in hypotheses.values() if h.get("confidence", 0) >= 0.75)
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total hypotheses: {total}")
    console.print(f"  [green]Confirmed: {confirmed}[/green]")
    console.print(f"  [yellow]High confidence (≥75%): {high_conf}[/yellow]")


@project.command()
@click.argument('name')
def path(name: str):
    """Get project directory path."""
    manager = ProjectManager()
    project_path = manager.get_project_path(name)
    
    if project_path:
        click.echo(project_path)
    else:
        console.print(f"[red]Project '{name}' not found.[/red]", err=True)
        raise click.Exit(1)


if __name__ == "__main__":
    project()