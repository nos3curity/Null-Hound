"""
File filtering command for Null-Hound.

Intelligently selects the most security-relevant files from a codebase using
heuristic scoring and LLM reranking.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


# --------- Heuristics (from whitelist.py) ---------

BIN_EXT = {
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico',
    '.pdf', '.zip', '.gz', '.tar', '.tgz', '.xz', '.7z', '.rar',
    '.woff', '.woff2', '.ttf', '.otf', '.eot', '.mp3', '.mp4', '.mov', '.webm',
}

DEFAULT_EXCLUDE_DIRS = {
    '.git', '.hg', '.svn', '.idea', '.vscode',
    'node_modules', 'vendor', 'dist', 'build', 'target', 'out', '.next', '.cache',
    '__pycache__', '.venv', 'venv', '.tox', '.mypy_cache', 'coverage', 'site-packages',
}

TEST_HINTS = re.compile(r"(^|/)(tests?|__tests__|testdata|spec|mocks?|fixtures?)(/|$)", re.I)
TEST_FILE_HINTS = re.compile(r"(test|spec|mock|fixture)\b", re.I)

LANG_WEIGHTS = {
    # Core languages
    '.c': 1.0, '.h': 0.9, '.cpp': 1.0, '.hpp': 0.9, '.cc': 1.0,
    '.go': 1.2, '.rs': 1.2, '.sol': 1.3, '.vy': 1.3, '.cairo': 1.3, '.move': 1.3,
    '.ts': 1.0, '.tsx': 1.0, '.js': 0.9, '.jsx': 0.9,
    '.py': 1.0, '.java': 0.9, '.cs': 0.8,
    # Config-ish (keep but with lower weight)
    '.toml': 0.5, '.json': 0.4, '.yml': 0.5, '.yaml': 0.5, '.ini': 0.4,
    '.md': 0.1,
}

PATH_BOOSTS = [
    (re.compile(r"(^|/)(src|lib|pkg|internal|core|server|api|cmd)(/|$)", re.I), 0.6),
    (re.compile(r"(^|/)(contracts?)(/|$)", re.I), 0.8),
    (re.compile(r"(^|/)(app|router|handler|controller|service|models?)(/|$)", re.I), 0.4),
]

PATH_PENALTIES = [
    (TEST_HINTS, -1.5),
    (re.compile(r"(^|/)(examples?|samples?|demos?|docs?)(/|$)", re.I), -0.6),
    (re.compile(r"(^|/)(scripts?|ci|\.github|doc)(/|$)", re.I), -0.3),
]

ENTRYPOINT_HINTS = [
    re.compile(r"(^|/)cmd/[^/]+/main\.go$", re.I),
    re.compile(r"(^|/)main\.go$", re.I),
    re.compile(r"(^|/)src/main\.rs$", re.I),
    re.compile(r"(^|/)src/main\.(ts|tsx|js|jsx)$", re.I),
]


@dataclass
class FileInfo:
    path: Path
    rel: str
    ext: str
    loc: int
    score: float
    reasons: list[str]


def is_binary(path: Path) -> bool:
    """Check if a file is binary."""
    if path.suffix.lower() in BIN_EXT:
        return True
    try:
        with open(path, 'rb') as f:
            chunk = f.read(1024)
        if b'\0' in chunk:
            return True
        # heuristic: too many non-text bytes
        text_chars = sum(c >= 9 and c <= 127 or c in (9, 10, 13) for c in chunk)
        return (text_chars / max(1, len(chunk))) < 0.80
    except Exception:
        return True


def count_loc(path: Path) -> int:
    """Count lines of code in a file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def score_file_with_preset(root: Path, path: Path, preset: dict, loader) -> tuple[float, list[str]]:
    """Return (score, reasons) for a file using preset configuration."""
    reasons: list[str] = []
    rel = str(path.relative_to(root))
    ext = path.suffix.lower()
    score = 0.0

    # Language weight from preset
    ext_weights = loader.get_extension_weights(preset)
    lw = ext_weights.get(ext, 0.2)
    score += lw
    reasons.append(f"lang:{ext}:{lw}")

    # Path boosts from preset
    for pattern, weight in loader.get_path_boosts(preset):
        rx = re.compile(pattern, re.I)
        if rx.search(rel):
            score += weight
            reasons.append(f"boost:{weight}")

    # Path penalties from preset
    for pattern, weight in loader.get_path_penalties(preset):
        rx = re.compile(pattern, re.I)
        if rx.search(rel):
            score += weight
            reasons.append(f"penalty:{weight}")

    # Tests or mock-like files get penalized extra
    base = path.name
    if TEST_FILE_HINTS.search(base):
        score -= 0.8
        reasons.append("testname:-0.8")

    # Entrypoints boost from preset
    for pattern in loader.get_entrypoint_patterns(preset):
        rx = re.compile(pattern, re.I)
        if rx.search(rel):
            score += 0.7
            reasons.append("entry:+0.7")
            break

    return score, reasons


def iter_candidates(root: Path, only_ext: set[str] | None = None) -> Iterable[Path]:
    """Iterate through candidate files in repository."""
    for p in root.rglob('*'):
        if p.is_dir():
            if p.name in DEFAULT_EXCLUDE_DIRS:
                continue
        elif p.is_file():
            rel = str(p.relative_to(root))
            # Exclude hidden files
            if any(seg.startswith('.') and seg not in {'.env'} for seg in p.parts):
                continue
            # Ignore obvious binaries
            if is_binary(p):
                continue
            # Only certain extensions if requested
            if only_ext is not None:
                if p.suffix.lower() not in only_ext:
                    continue
            yield p


def build_candidates_from_preset(root: Path, preset: dict, loader, workers: int = 8) -> list[FileInfo]:
    """Build list of candidate files with scores using preset configuration."""
    # Get extensions from preset
    ext_weights = loader.get_extension_weights(preset)
    only_ext = set(ext_weights.keys())

    paths = list(iter_candidates(root, only_ext=only_ext))
    items: list[FileInfo] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Scanning {len(paths)} files...", total=len(paths))

        with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
            futs = {ex.submit(count_loc, p): p for p in paths}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    loc = fut.result()
                except Exception:
                    loc = 0
                if loc <= 0:
                    progress.advance(task)
                    continue
                score, reasons = score_file_with_preset(root, p, preset, loader)
                items.append(FileInfo(
                    path=p,
                    rel=str(p.relative_to(root)),
                    ext=p.suffix.lower(),
                    loc=loc,
                    score=score,
                    reasons=reasons,
                ))
                progress.advance(task)

    return items


def heuristic_rank(items: list[FileInfo]) -> list[FileInfo]:
    """Rank files by heuristic score."""
    return sorted(items, key=lambda x: (x.score, min(x.loc, 5000)), reverse=True)


def llm_rerank(items: list[FileInfo], preset: dict, loader, config: dict, model: str = 'gemini-2.5-pro', max_items: int = 300) -> list[str]:
    """Use LLM to rerank candidate files; returns list of relpaths in prioritized order.

    Uses the project's UnifiedLLMClient which includes retry/backoff logic.
    """
    from llm.unified_client import UnifiedLLMClient
    from pydantic import BaseModel, Field

    # Define response schema
    class FilterResponse(BaseModel):
        prioritized: list[str] = Field(description="Array of relpaths in priority order")

    # Prepare compact JSON describing candidates
    sample = [
        {
            'rel': it.rel,
            'loc': it.loc,
            'score': round(it.score, 3),
            'ext': it.ext,
            'reasons': it.reasons[:4],
        }
        for it in items[:max_items]
    ]

    # Build system prompt with preset-specific security focus
    security_focus = loader.get_security_focus(preset)
    system_prompt = (
        f"You prioritize source files for a security audit whitelist. {security_focus}"
    )

    # Hardcoded response format instructions
    user_content = json.dumps({
        'candidates': sample,
        'instructions': (
            'Return JSON with a single field "prioritized", an array of relpaths, in priority order. '
            'Do not include any other fields or explanations.'
        )
    }, indent=2)

    try:
        with console.status("[cyan]Reranking with LLM...", spinner="dots"):
            t0 = time.time()

            # Create a minimal config for the filter profile
            # This ensures we use the project's retry/backoff logic
            filter_config = {
                **config,
                "models": {
                    "filter": {
                        "provider": "gemini",
                        "model": model
                    }
                }
            }

            # Use UnifiedLLMClient with retry/backoff support
            client = UnifiedLLMClient(filter_config, profile="filter")
            response = client.parse(
                system=system_prompt,
                user=user_content,
                schema=FilterResponse
            )

            console.print(f"[green]✓ Rerank completed in {time.time()-t0:.1f}s[/green]")

        out = response.prioritized
        if out:
            return out
        return [it.rel for it in items]
    except Exception as e:
        console.print(f"[red]LLM rerank failed: {e}[/red]")
        console.print("[yellow]Falling back to heuristic ranking[/yellow]")
        return [it.rel for it in items]


def compile_whitelist(prioritized_paths: list[str], by_rel: dict[str, FileInfo], limit_loc: int) -> tuple[list[str], int]:
    """Compile final whitelist respecting LOC limit."""
    selected: list[str] = []
    total = 0
    for rel in prioritized_paths:
        info = by_rel.get(rel)
        if not info:
            continue
        if total + info.loc > limit_loc and total > 0:
            continue
        selected.append(rel)
        total += info.loc
        if total >= limit_loc:
            break
    return selected, total


@click.command()
@click.argument('project_name')
@click.option('--limit-loc', type=int, default=40000, help='Total LOC budget for filter')
@click.option('--model', default='gemini-2.5-pro', help='Gemini model to use')
@click.option('--max-llm-items', type=int, default=300, help='Max candidates to send to LLM')
@click.option('--verbose', is_flag=True, help='Verbose output')
def filter_files(project_name: str, limit_loc: int, model: str, max_llm_items: int, verbose: bool):
    """Generate a filtered file list optimized for security analysis.

    Uses preset-configured heuristics and LLM reranking to select the most
    security-relevant files from a codebase, respecting a LOC budget.

    The filter is saved to the project's filters directory.
    """
    from commands.project import ProjectManager
    from utils.presets import get_preset_loader
    from utils.config_loader import load_config

    # Load project config (for LLM client)
    config = load_config()

    # Load project
    manager = ProjectManager()
    proj = manager.get_project(project_name)
    if not proj:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        raise click.Abort()

    project_path = manager.get_project_path(project_name)

    # Load project config to get source path and preset
    config_file = project_path / "project.json"
    if not config_file.exists():
        console.print(f"[red]Project config not found: {config_file}[/red]")
        raise click.Abort()

    with open(config_file) as f:
        project_config = json.load(f)

    source_path = project_config.get("source_path")
    preset_name = project_config.get("preset", "default")

    if not source_path:
        console.print(f"[red]Project '{project_name}' has no source_path[/red]")
        raise click.Abort()

    root = Path(source_path).resolve()
    if not root.exists() or not root.is_dir():
        console.print(f"[red]Source path not found or not a directory: {root}[/red]")
        raise click.Abort()

    # Load preset
    loader = get_preset_loader()
    try:
        preset = loader.load(preset_name)
    except Exception as e:
        console.print(f"[red]Failed to load preset '{preset_name}': {e}[/red]")
        raise click.Abort()

    console.print(f"[cyan]Using preset:[/cyan] {preset_name}")
    console.print(f"[cyan]Source path:[/cyan] {root}")

    # Get extension filter from preset (only include files with these extensions)
    preset_extensions = set(loader.get_extension_weights(preset).keys())
    if verbose:
        console.print(f"[dim]Preset extensions: {', '.join(sorted(preset_extensions))}[/dim]")

    # Build candidates using preset configuration
    items = build_candidates_from_preset(root, preset, loader, workers=os.cpu_count() or 8)
    if not items:
        console.print("[yellow]No candidate files found.[/yellow]")
        raise click.Abort()

    console.print(f"[green]Found {len(items)} candidate files[/green]")

    # Heuristic ranking
    ranked = heuristic_rank(items)
    by_rel = {it.rel: it for it in ranked}

    # LLM reranking (always enabled)
    if verbose:
        console.print(f"[cyan]Reranking top {min(len(ranked), max_llm_items)} candidates with {model}...[/cyan]")

    prioritized = llm_rerank(ranked, preset, loader, config, model=model, max_items=max_llm_items)

    # Keep only those we have, in that order, then append unseen from heuristic
    seen = set()
    pruned = []
    for rel in prioritized:
        if rel in by_rel and rel not in seen:
            pruned.append(rel)
            seen.add(rel)
    for it in ranked:
        if it.rel not in seen:
            pruned.append(it.rel)
            seen.add(it.rel)
    prioritized = pruned

    # Compile final list
    selected, total = compile_whitelist(prioritized, by_rel, limit_loc)

    # Write output to project's filters directory (overwrite existing)
    filters_dir = project_path / "filters"
    filters_dir.mkdir(exist_ok=True)
    out_path = filters_dir / "filter.txt"
    out_text = '\n'.join(selected) + ('\n' if selected else '')
    out_path.write_text(out_text, encoding='utf-8')

    # Summary
    console.print(f"\n[bold green]✓ Wrote {len(selected)} files ({total:,} LOC) to {out_path}[/bold green]")

    # Extension breakdown
    by_ext: dict[str, int] = {}
    for rel in selected:
        ext = Path(rel).suffix.lower() or '(no ext)'
        by_ext[ext] = by_ext.get(ext, 0) + 1

    table = Table(title="Selected Files by Extension", show_header=True)
    table.add_column("Extension", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("LOC", justify="right", style="yellow")

    for ext, cnt in sorted(by_ext.items(), key=lambda x: (-x[1], x[0])):
        ext_loc = sum(by_rel[rel].loc for rel in selected if (Path(rel).suffix.lower() or '(no ext)') == ext)
        table.add_row(ext, str(cnt), f"{ext_loc:,}")

    console.print(table)

    if verbose:
        console.print(f"\n[dim]Top 10 selected files:[/dim]")
        for i, rel in enumerate(selected[:10], 1):
            info = by_rel.get(rel)
            if info:
                console.print(f"  {i:2d}. {rel} ({info.loc} LOC, score={info.score:.2f})")