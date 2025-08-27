"""
Generate professional security audit reports from project analysis.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel

from commands.project import ProjectManager
from analysis.report_generator import ReportGenerator

console = Console()


@click.command()
@click.argument('project_name')
@click.option('--output', '-o', help="Output file path (default: project_dir/reports/audit_report_TIMESTAMP.html)")
@click.option('--format', '-f', type=click.Choice(['html', 'markdown', 'pdf']), default='html', help="Report format")
@click.option('--title', '-t', help="Custom report title")
@click.option('--auditors', '-a', help="Comma-separated list of auditor names", default="Security Team")
@click.option('--debug', is_flag=True, help="Enable debug mode")
@click.option('--show-prompt', is_flag=True, help="Show the LLM prompt and response used to generate the report")
def report(project_name: str, output: Optional[str], format: str, 
          title: Optional[str], auditors: str, debug: bool, show_prompt: bool):
    """
    Generate a professional security audit report for a project.
    
    Creates a comprehensive report including:
    - Executive summary
    - Findings (when available)
    - Scope and methodology
    - Testing coverage appendix
    """
    manager = ProjectManager()
    project = manager.get_project(project_name)
    
    if not project:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        raise click.Exit(1)
    
    project_dir = Path(project["path"])
    
    # Check for required data
    graphs_dir = project_dir / "graphs"
    if not graphs_dir.exists() or not list(graphs_dir.glob("*.json")):
        console.print("[red]No graphs found. Run graph build first.[/red]")
        raise click.Exit(1)
    
    # Load hypotheses if available
    hypothesis_file = project_dir / "hypotheses.json"
    hypotheses = {}
    if hypothesis_file.exists():
        with open(hypothesis_file, 'r') as f:
            hyp_data = json.load(f)
            hypotheses = hyp_data.get("hypotheses", {})
    
    # Determine output path
    if not output:
        reports_dir = project_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = reports_dir / f"audit_report_{timestamp}.{format}"
    else:
        output_path = Path(output)
    
    console.print(Panel(
        f"[bold cyan]Generating Security Audit Report[/bold cyan]\n\n"
        f"[bold]Project:[/bold] {project_name}\n"
        f"[bold]Format:[/bold] {format.upper()}\n"
        f"[bold]Output:[/bold] {output_path.name}\n"
        f"[bold]Hypotheses Tested:[/bold] {len(hypotheses)}",
        title="[bold]Report Generation[/bold]",
        border_style="cyan"
    ))
    
    # Initialize report generator
    from commands.graph import load_config
    config = load_config()
    
    generator = ReportGenerator(
        project_dir=project_dir,
        config=config,
        debug=debug
    )
    
    # Progress callback from generator
    def _progress_cb(ev: Dict):
        status = ev.get('status', '')
        msg = ev.get('message', '')
        if status in ('start',):
            console.print(f"[cyan]🚀 Booting report engines...[/cyan]")
        elif status in ('llm',):
            console.print(f"[cyan]🧠 Cooking up summary + overview...[/cyan]")
        elif status in ('llm_done',):
            console.print(f"[green]✅ Summary + overview ready[/green]")
        elif status in ('findings',):
            console.print(f"[cyan]🔍 Hunting confirmed findings...[/cyan]")
        elif status in ('findings_describe',):
            console.print(f"[cyan]✍️  Polishing finding write-ups...[/cyan]")
        elif status in ('snippets',):
            console.print(f"[cyan]🧩 Picking code bites: {msg}[/cyan]")
        elif status in ('snippets_done',):
            console.print(f"[green]✅ {msg}[/green]")
        elif status in ('render',):
            console.print(f"[cyan]🖨️  Forging the final scroll...[/cyan]")
        elif status in ('findings_done',):
            console.print(f"[green]✅ {msg}[/green]")
        else:
            # Generic fallback
            if msg:
                console.print(f"[dim]{msg}[/dim]")
    
    try:
        report_data = generator.generate(
            project_name=project_name,
            project_source=project["source_path"],
            title=title or f"Security Audit: {project_name}",
            auditors=auditors.split(','),
            format=format,
            progress_callback=_progress_cb
        )
        
        # Optionally show prompt/response for debugging
        if show_prompt:
            try:
                from rich.syntax import Syntax
                if generator.last_prompt:
                    console.print(Panel(Syntax(generator.last_prompt, "json", theme="monokai", word_wrap=True), title="Prompt"))
                if generator.last_response:
                    console.print(Panel(Syntax(generator.last_response, "json", theme="monokai", word_wrap=True), title="Raw Response"))
            except Exception:
                # Fall back to plain text
                if generator.last_prompt:
                    console.print(Panel(generator.last_prompt, title="Prompt"))
                if generator.last_response:
                    console.print(Panel(generator.last_response, title="Raw Response"))

        # Write report
        console.print(f"[cyan]Writing {format.upper()} report...[/cyan]")
        
        if format == 'html':
            with open(output_path, 'w') as f:
                f.write(report_data)
        elif format == 'markdown':
            with open(output_path, 'w') as f:
                f.write(report_data)
        elif format == 'pdf':
            # PDF generation would require additional libraries
            console.print("[yellow]PDF generation not yet implemented. Generating HTML instead.[/yellow]")
            output_path = output_path.with_suffix('.html')
            with open(output_path, 'w') as f:
                f.write(report_data)
        
        console.print(f"[green]✓ Report generated successfully![/green]")
        console.print(f"[green]Location: {output_path}[/green]")
        
        # Try to open in browser if HTML
        if format == 'html':
            import webbrowser
            try:
                webbrowser.open(f"file://{output_path.absolute()}")
                console.print("[dim]Report opened in browser[/dim]")
            except:
                pass
                
    except Exception as e:
        console.print(f"[red]Report generation failed: {e}[/red]")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        raise click.Exit(1)


if __name__ == "__main__":
    report()
