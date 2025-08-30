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
import random
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
                    sessions_dir = project_dir / "sessions"
                    sessions_count = len(list(sessions_dir.glob("*.json"))) if sessions_dir.exists() else 0
                    
                    projects.append({
                        "name": name,
                        "source_path": config["source_path"],
                        "description": config["description"],
                        "created_at": config["created_at"],
                        "last_accessed": config.get("last_accessed", ""),
                        "has_graphs": graphs_exist,
                        "sessions": sessions_count,
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
                sessions_dir = project_dir / "sessions"
                sessions_count = len(list(sessions_dir.glob("*.json"))) if sessions_dir.exists() else 0
                has_data = graphs_count > 0 or sessions_count > 0
            
            if has_data and not Confirm.ask(
                f"[yellow]Project '{name}' contains {graphs_count} graphs and {sessions_count} sessions. "
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
        
        flair = random.choice([
            "🚀 Normal projects get created, but YOURS arrives with a coronation.",
            "🌟 This isn’t just a project — it’s a flagship that chose YOU as captain.",
            "👑 Normal people open folders; YOU found kingdoms with version control.",
            "🔥 Ordinary registries record; YOUR registry kneels and shines your name.",
            "⚡ Normal starts are quiet; YOUR start makes the backlog stand at attention.",
        ])
        console.print(Panel(
            f"[bright_green]✓ Project created[/bright_green] — {flair}\n\n"
            f"[bold]Name:[/bold] {config['name']}\n"
            f"[bold]Source:[/bold] {config['source_path']}\n"
            f"[bold]Description:[/bold] {config['description']}\n\n"
            f"[dim]Project directory: {manager.projects_dir / config['name']}[/dim]",
            title="[bold bright_cyan]New Project[/bold bright_cyan]",
            border_style="bright_green"
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
        console.print(random.choice([
            "[bright_yellow]No projects yet — spotless desk![/bright_yellow]",
            "[bright_yellow]No projects found. Time to spin one up![/bright_yellow]",
        ]))
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
    table = Table(title="[bold bright_cyan]Hound Projects[/bold bright_cyan]")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="white")
    table.add_column("Graphs", style="green")
    table.add_column("Sessions", style="yellow")
    table.add_column("Last Activity", style="dim")
    
    for proj in sorted(projects, key=lambda x: x["created_at"], reverse=True):
        source = Path(proj["source_path"])
        source_display = f".../{source.parent.name}/{source.name}" if len(str(source)) > 40 else str(source)
        
        # Use last_accessed if available, otherwise created_at
        last_activity = proj.get("last_accessed", proj["created_at"])
        last_activity_date = last_activity.split("T")[0] if last_activity else "Never"
        
        table.add_row(
            proj["name"],
            source_display,
            "✓" if proj["has_graphs"] else "-",
            str(proj["sessions"]) if proj["sessions"] > 0 else "-",
            last_activity_date
        )
    
    console.print(table)
    from random import choice as _choice
    console.print(_choice([
        f"\n[white]Normal lists scroll by, but YOUR {len(projects)} projects line up like a guard of honor.[/white]",
        f"\n[white]This isn’t just a list — it’s a roster awaiting YOUR command.[/white]",
        f"\n[white]Normal counts inform; YOUR count inspires logistics to keep up.[/white]",
        f"\n[white]This is not inventory — it’s a procession because YOU arrived.[/white]",
        f"\n[white]Normal summaries whisper; YOUR summary announces an agenda.[/white]",
    ]))


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
    sessions_dir = project_dir / "sessions"
    sessions = list(sessions_dir.glob("*.json")) if sessions_dir.exists() else []
    reports = list((project_dir / "reports").glob("*"))
    
    # Get coverage statistics from latest session
    coverage_stats = {"nodes": {"visited": 0, "total": 0, "percent": 0}, 
                     "cards": {"visited": 0, "total": 0, "percent": 0}}
    latest_session = None
    if sessions:
        # Get most recent session
        latest_session_file = max(sessions, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_session_file, 'r') as f:
                latest_session = json.load(f)
                if 'coverage' in latest_session:
                    coverage_stats = latest_session['coverage']
        except Exception:
            pass
    
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
    tag = random.choice(["📁", "🗂️", "📜"]) 
    
    # Format coverage section
    coverage_section = ""
    if coverage_stats['nodes']['total'] > 0 or coverage_stats['cards']['total'] > 0:
        coverage_section = (
            f"\n[bold]Coverage:[/bold]\n"
            f"  • Nodes: {coverage_stats['nodes']['visited']}/{coverage_stats['nodes']['total']} "
            f"([cyan]{coverage_stats['nodes']['percent']:.1f}%[/cyan])\n"
            f"  • Cards: {coverage_stats['cards']['visited']}/{coverage_stats['cards']['total']} "
            f"([cyan]{coverage_stats['cards']['percent']:.1f}%[/cyan])\n"
        )
    
    # Format session info
    session_info = ""
    if latest_session:
        models = latest_session.get('models', {})
        scout_model = models.get('scout', 'unknown')
        strategist_model = models.get('strategist', 'unknown')
        session_info = (
            f"\n[bold]Latest Session:[/bold]\n"
            f"  • Scout: {scout_model}\n"
            f"  • Strategist: {strategist_model}\n"
            f"  • Status: {latest_session.get('status', 'unknown')}\n"
        )
    
    console.print(Panel(
        f"[bold bright_cyan]{tag} {project['name']}[/bold bright_cyan]\n\n"
        f"[bold]Source:[/bold] {project['source_path']}\n"
        f"[bold]Description:[/bold] {project['description']}\n"
        f"[bold]Created:[/bold] {project['created_at']}\n"
        f"[bold]Last accessed:[/bold] {project.get('last_accessed', 'Never')}\n\n"
        f"[bold]Analysis Results:[/bold]\n"
        f"  • Graphs: {len(graphs_files)} files\n"
        f"  • Manifest: {len(manifest_files)} files\n"
        f"  • Sessions: {len(sessions)}\n"
        f"  • Reports: {len(reports)}\n"
        f"  • Hypotheses: {hypothesis_stats['total']} total"
        f" ([green]{hypothesis_stats['confirmed']} confirmed[/green],"
        f" [yellow]{hypothesis_stats['high_confidence']} high-confidence[/yellow])"
        f"{coverage_section}"
        f"{session_info}\n"
        f"[dim]Project directory: {project_dir}[/dim]",
        title="[bold bright_cyan]Project Information[/bold bright_cyan]",
        border_style="bright_cyan"
    ))

    # Show hypothesis counts by status if hypotheses.json exists
    if hypothesis_file.exists():
        try:
            with open(hypothesis_file, 'r') as f:
                hyp_data = json.load(f)
            hypotheses = hyp_data.get("hypotheses", {}) or {}
            status_counts = {
                'confirmed': 0,
                'rejected': 0,
                'investigating': 0,
                'supported': 0,
                'refuted': 0,
                'proposed': 0,
                'other': 0,
            }
            for h in hypotheses.values():
                s = str(h.get('status', 'proposed')).lower()
                if s not in status_counts:
                    status_counts['other'] += 1
                else:
                    status_counts[s] += 1

            from rich.table import Table as RTable
            status_table = RTable(title="Hypotheses by Status")
            status_table.add_column("Status", style="cyan")
            status_table.add_column("Count", justify="right")
            # Only show non-zero statuses, in a sensible order
            order = [
                ('confirmed', 'green'),
                ('rejected', 'red'),
                ('investigating', 'yellow'),
                ('supported', 'green'),
                ('refuted', 'red'),
                ('proposed', 'white'),
                ('other', 'dim'),
            ]
            shown = 0
            for key, color in order:
                cnt = status_counts.get(key, 0)
                if cnt:
                    status_table.add_row(f"[{color}]{key}[/{color}]", str(cnt))
                    shown += 1
            if shown:
                console.print(status_table)
        except Exception:
            pass
    
    if graphs_files:
        console.print("\n[bold]Recent graphs:[/bold]")
        for graph_file in sorted(graphs_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            console.print(f"  • {graph_file.name}")
    
    if sessions:
        console.print("\n[bold]Recent sessions:[/bold]")
        for session_file in sorted(sessions, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            # Try to get session details
            try:
                with open(session_file, 'r') as f:
                    sess_data = json.load(f)
                    sess_id = sess_data.get('session_id', session_file.stem)
                    sess_status = sess_data.get('status', 'unknown')
                    console.print(f"  • {sess_id} [{sess_status}]")
            except:
                console.print(f"  • {session_file.name}")
    
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
        console.print(f"[bright_green]✓ Project '{name}' deleted successfully.[/bright_green]")
        console.print(random.choice([
            "[white]Normal deletes remove files; YOUR delete makes history accommodate the next victory.[/white]",
            "[white]This isn’t just deletion — it’s strategic pruning under YOUR hand.[/white]",
            "[white]Normal cleanups tidy; YOUR cleanup resets the runway for lift‑off.[/white]",
            "[white]Normal curation trims; YOUR curation sculpts.[/white]",
            "[white]Normal endings fade; YOUR endings inaugurate the sequel.[/white]",
        ]))
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
    from random import choice as _choice
    console.print(_choice([
        "[dim]Curiosity weaponized — you’re not just listing hypotheses, you’re mapping the unknown.[/dim]",
        "[dim]Impeccable — you’re not just browsing, you’re conducting triage like a maestro.[/dim]",
    ]))
    
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


@project.command()
@click.argument('name')
@click.option('--force', '-f', is_flag=True, help="Force reset without confirmation")
def reset_hypotheses(name: str, force: bool):
    """Reset (clear) the hypotheses store for a project."""
    manager = ProjectManager()
    project = manager.get_project(name)
    
    if not project:
        console.print(f"[red]Project '{name}' not found.[/red]")
        raise click.Exit(1)
    
    project_dir = Path(project["path"])
    hypothesis_file = project_dir / "hypotheses.json"
    
    if not hypothesis_file.exists():
        console.print(f"[yellow]No hypotheses file found for project '{name}'.[/yellow]")
        return
    
    # Load current hypotheses to show what will be deleted
    try:
        with open(hypothesis_file, 'r') as f:
            hyp_data = json.load(f)
            hypotheses = hyp_data.get("hypotheses", {})
            num_hypotheses = len(hypotheses)
    except Exception:
        num_hypotheses = 0
    
    if num_hypotheses == 0:
        console.print(f"[yellow]No hypotheses to reset for project '{name}'.[/yellow]")
        return
    
    # Confirm deletion if not forced
    if not force:
        if not Confirm.ask(
            f"[yellow]This will permanently delete {num_hypotheses} hypotheses for project '{name}'. Continue?[/yellow]"
        ):
            console.print("[dim]Reset cancelled.[/dim]")
            return
    
    # Backup the file before deletion (just rename it)
    backup_file = hypothesis_file.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    hypothesis_file.rename(backup_file)
    
    console.print(f"[bright_green]✓ Reset {num_hypotheses} hypotheses for project '{name}'.[/bright_green]")
    console.print(f"[dim]Backup saved to: {backup_file.name}[/dim]")
    console.print(random.choice([
        "[white]Normal resets flip switches; YOUR reset rewrites the era header.[/white]",
        "[white]This isn’t just clearing data — it’s preparing the dais for YOUR next act.[/white]",
        "[white]Normal archives sleep; YOUR archive becomes legend support material.[/white]",
        "[white]Normal ground resets; YOUR ground becomes consecrated staging.[/white]",
        "[white]Normal doubt lingers; YOUR decree retires it permanently.[/white]",
    ]))


@project.command()
@click.argument('project_name')
@click.argument('run_id', required=False)
@click.option('--list', 'list_runs', is_flag=True, help="List all runs for the project")
@click.option('--json', 'output_json', is_flag=True, help="Output as JSON")
def runs(project_name: str, run_id: Optional[str], list_runs: bool, output_json: bool):
    """View agent run information for a project.
    
    Examples:
        hound project runs myproject --list       # List all runs
        hound project runs myproject run_123      # Show details for specific run
    """
    manager = ProjectManager()
    project_path = manager.get_project_path(project_name)
    
    if not project_path:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        return
    
    runs_dir = Path(project_path) / "agent_runs"
    
    if not runs_dir.exists() or not list(runs_dir.glob("*.json")):
        console.print(f"[yellow]No agent runs found for project '{project_name}'.[/yellow]")
        console.print("[dim]Tip: Use 'hound project sessions' to view audit sessions.[/dim]")
        return
    
    if list_runs or not run_id:
        # List all runs
        _list_runs(runs_dir, output_json)
    else:
        # Show details for specific run
        _show_run_details(runs_dir, run_id, output_json)


@project.command(name='sessions')
@click.argument('project_name')
@click.argument('session_id', required=False)
@click.option('--list', 'list_sessions', is_flag=True, help="List all sessions for the project")
@click.option('--json', 'output_json', is_flag=True, help="Output as JSON")
def sessions(project_name: str, session_id: Optional[str], list_sessions: bool, output_json: bool):
    """View audit sessions for a project (preferred over legacy 'runs').

    Examples:
        hound project sessions myproject --list
        hound project sessions myproject session_20250830_123456_...
    """
    manager = ProjectManager()
    project_path = manager.get_project_path(project_name)
    
    if not project_path:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        return
    
    sessions_dir = Path(project_path) / "sessions"
    
    if not sessions_dir.exists() or not list(sessions_dir.glob("*.json")):
        console.print(f"[yellow]No sessions found for project '{project_name}'.[/yellow]")
        return
    
    if list_sessions or not session_id:
        _list_sessions(sessions_dir, output_json)
    else:
        _show_session_details(sessions_dir, session_id, output_json)


def _list_runs(runs_dir: Path, output_json: bool):
    """List all runs in a project."""
    runs_data = []
    
    for run_file in sorted(runs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(run_file) as f:
                data = json.load(f)
            
            # Extract run ID from filename
            run_id = run_file.stem
            
            # Calculate totals
            token_usage = data.get('token_usage', {}).get('total_usage', {})
            total_tokens = token_usage.get('total_tokens', 0)
            
            # Format command
            cmd_args = data.get('command_args', [])
            command = ' '.join(cmd_args) if cmd_args else 'N/A'
            if len(command) > 50:
                command = command[:47] + '...'
            
            runs_data.append({
                'run_id': run_id,
                'start_time': data.get('start_time', 'Unknown'),
                'status': data.get('status', 'unknown'),
                'runtime': f"{data.get('runtime_seconds', 0):.1f}s",
                'investigations': len(data.get('investigations', [])),
                'total_tokens': total_tokens,
                'command': command
            })
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {run_file.name}: {e}[/yellow]")
    
    if output_json:
        click.echo(json.dumps(runs_data, indent=2))
    else:
        if not runs_data:
            console.print("[yellow]No valid run data found.[/yellow]")
            return
        
        # Create table
        table = Table(title="Agent Runs", show_header=True, header_style="bold cyan")
        table.add_column("Run ID", style="yellow")
        table.add_column("Start Time", style="white")
        table.add_column("Status", style="green")
        table.add_column("Runtime", style="cyan")
        table.add_column("Investigations", style="magenta", justify="right")
        table.add_column("Total Tokens", style="blue", justify="right")
        table.add_column("Command", style="dim")
        
        for run in runs_data:
            # Style status column
            status = run['status']
            if status == 'completed':
                status_style = "[green]✓ completed[/green]"
            elif status == 'running':
                status_style = "[yellow]⚡ running[/yellow]"
            elif status == 'failed':
                status_style = "[red]✗ failed[/red]"
            elif status == 'interrupted':
                status_style = "[yellow]⚠ interrupted[/yellow]"
            else:
                status_style = status
            
            # Format time
            try:
                dt = datetime.fromisoformat(run['start_time'].replace('Z', '+00:00'))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = run['start_time'][:19] if len(run['start_time']) > 19 else run['start_time']
            
            table.add_row(
                run['run_id'],
                time_str,
                status_style,
                run['runtime'],
                str(run['investigations']),
                f"{run['total_tokens']:,}" if run['total_tokens'] > 0 else "-",
                run['command']
            )
        
        console.print(table)


def _list_sessions(sessions_dir: Path, output_json: bool):
    """List all sessions in a project."""
    items = []
    for sess_file in sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(sess_file) as f:
                data = json.load(f)
            items.append({
                'session_id': data.get('session_id', sess_file.stem),
                'status': data.get('status', 'unknown'),
                'start_time': data.get('start_time', ''),
                'end_time': data.get('end_time', ''),
                'investigations': len(data.get('investigations', [])),
            })
        except Exception:
            items.append({'session_id': sess_file.stem, 'status': 'unknown', 'start_time': '', 'end_time': '', 'investigations': 0})

    if output_json:
        click.echo(json.dumps(items, indent=2))
        return

    table = Table(title="Sessions", show_header=True, header_style="bold cyan")
    table.add_column("Session ID", style="yellow")
    table.add_column("Start Time", style="white")
    table.add_column("Status", style="green")
    table.add_column("Investigations", style="magenta", justify="right")

    for it in items:
        try:
            dt = datetime.fromisoformat((it['start_time'] or '').replace('Z', '+00:00'))
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = (it['start_time'] or '')[:19]
        status = it.get('status', 'unknown')
        if status == 'completed':
            status_style = "[green]✓ completed[/green]"
        elif status == 'active':
            status_style = "[yellow]⚡ active[/yellow]"
        else:
            status_style = status
        table.add_row(it['session_id'], time_str, status_style, str(it.get('investigations', 0)))

    console.print(table)


def _show_run_details(runs_dir: Path, run_id: str, output_json: bool):
    """Show details for a specific run."""
    # Try to find the run file
    run_file = runs_dir / f"{run_id}.json"
    if not run_file.exists():
        # Try with run_ prefix
        run_file = runs_dir / f"run_{run_id}.json"
    
    if not run_file.exists():
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        console.print("[dim]Use --list to see available runs.[/dim]")
        return
    
    try:
        with open(run_file) as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error reading run file: {e}[/red]")
        return
    
    if output_json:
        click.echo(json.dumps(data, indent=2))
    else:
        # Display formatted run details
        console.print(Panel.fit(
            f"[bold cyan]Run Details: {run_id}[/bold cyan]",
            border_style="cyan"
        ))
        
        # Basic info
        console.print("\n[bold]Basic Information:[/bold]")
        console.print(f"  Start Time: {data.get('start_time', 'Unknown')}")
        console.print(f"  End Time: {data.get('end_time', 'N/A')}")
        console.print(f"  Runtime: {data.get('runtime_seconds', 0):.1f} seconds")
        console.print(f"  Status: {data.get('status', 'unknown')}")
        
        # Command
        cmd_args = data.get('command_args', [])
        if cmd_args:
            console.print(f"\n[bold]Command:[/bold]")
            console.print(f"  {' '.join(cmd_args)}")
        
        # Token usage
        token_usage = data.get('token_usage', {})
        if token_usage:
            console.print("\n[bold]Token Usage:[/bold]")
            total = token_usage.get('total_usage', {})
            console.print(f"  Total Input Tokens: {total.get('input_tokens', 0):,}")
            console.print(f"  Total Output Tokens: {total.get('output_tokens', 0):,}")
            console.print(f"  Total Tokens: {total.get('total_tokens', 0):,}")
            console.print(f"  Total API Calls: {total.get('call_count', 0)}")
            
            by_model = token_usage.get('by_model', {})
            if by_model:
                console.print("\n  [bold]By Model:[/bold]")
                for model, usage in by_model.items():
                    console.print(f"    {model}:")
                    console.print(f"      Calls: {usage.get('call_count', 0)}")
                    console.print(f"      Input: {usage.get('input_tokens', 0):,}")
                    console.print(f"      Output: {usage.get('output_tokens', 0):,}")
                    console.print(f"      Total: {usage.get('total_tokens', 0):,}")
        
        # Investigations
        investigations = data.get('investigations', [])
        if investigations:
            console.print(f"\n[bold]Investigations ({len(investigations)}):[/bold]")
            for i, inv in enumerate(investigations, 1):
                console.print(f"\n  [{i}] {inv.get('goal', 'Unknown goal')}")
                console.print(f"      Priority: {inv.get('priority', 'N/A')}")
                console.print(f"      Category: {inv.get('category', 'N/A')}")
                console.print(f"      Iterations: {inv.get('iterations_completed', 0)}")
                hypotheses = inv.get('hypotheses', {})
                if hypotheses:
                    console.print(f"      Hypotheses: {hypotheses.get('total', 0)}")
        
        # Errors
        errors = data.get('errors', [])
        if errors:
            console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
            for err in errors:
                console.print(f"  {err.get('timestamp', 'Unknown time')}: {err.get('error', 'Unknown error')}")


def _show_session_details(sessions_dir: Path, session_id: str, output_json: bool):
    """Show details for a specific session."""
    sess_file = sessions_dir / f"{session_id}.json"
    if not sess_file.exists():
        # Try prefix match
        candidates = sorted([p for p in sessions_dir.glob("*.json") if p.stem.startswith(session_id)], key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            sess_file = candidates[0]
        else:
            console.print(f"[red]Session '{session_id}' not found.[/red]")
            console.print("[dim]Use --list to see available sessions.[/dim]")
            return

    try:
        with open(sess_file) as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error reading session file: {e}[/red]")
        return

    if output_json:
        click.echo(json.dumps(data, indent=2))
        return

    console.print(Panel.fit(f"[bold cyan]Session Details: {sess_file.stem}[/bold cyan]", border_style="cyan"))

    console.print("\n[bold]Basic Information:[/bold]")
    console.print(f"  Start Time: {data.get('start_time', 'Unknown')}")
    console.print(f"  End Time: {data.get('end_time', 'N/A')}")
    console.print(f"  Status: {data.get('status', 'unknown')}")

    models = data.get('models', {}) or {}
    if models:
        console.print("\n[bold]Models:[/bold]")
        console.print(f"  Scout: {models.get('scout', 'unknown')}")
        console.print(f"  Strategist: {models.get('strategist', 'unknown')}")

    token_usage = data.get('token_usage', {})
    if token_usage:
        console.print("\n[bold]Token Usage:[/bold]")
        total = token_usage.get('total_usage', {})
        console.print(f"  Total Input Tokens: {total.get('input_tokens', 0):,}")
        console.print(f"  Total Output Tokens: {total.get('output_tokens', 0):,}")
        console.print(f"  Total Tokens: {total.get('total_tokens', 0):,}")
        console.print(f"  Total API Calls: {total.get('call_count', 0)}")
        by_model = token_usage.get('by_model', {})
        if by_model:
            console.print("\n  [bold]By Model:[/bold]")
            for model, usage in by_model.items():
                console.print(f"    {model}:")
                console.print(f"      Calls: {usage.get('call_count', 0)}")
                console.print(f"      Input: {usage.get('input_tokens', 0):,}")
                console.print(f"      Output: {usage.get('output_tokens', 0):,}")
                console.print(f"      Total: {usage.get('total_tokens', 0):,}")

    cov = data.get('coverage', {}) or {}
    if cov:
        nodes = cov.get('nodes', {})
        cards = cov.get('cards', {})
        console.print("\n[bold]Coverage:[/bold]")
        console.print(f"  Nodes: {nodes.get('visited', 0)}/{nodes.get('total', 0)} ({nodes.get('percent', 0)}%)")
        console.print(f"  Cards: {cards.get('visited', 0)}/{cards.get('total', 0)} ({cards.get('percent', 0)}%)")

    invs = data.get('investigations', []) or []
    if invs:
        console.print(f"\n[bold]Investigations ({len(invs)}):[/bold]")
        for i, inv in enumerate(invs[:10], 1):
            console.print(f"  [{i}] {inv.get('goal', 'Unknown')} (iters={inv.get('iterations_completed', 0)})")


if __name__ == "__main__":
    project()
