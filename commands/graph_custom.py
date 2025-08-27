"""Custom graph builder that reuses the main graph building logic."""

import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.graph_builder import GraphBuilder
import random
from llm.client import LLMClient
from pydantic import BaseModel, Field
from typing import List as ListType
from commands.graph import load_config


class CustomGraphSpec(BaseModel):
    """Specification for a custom graph"""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Graph name")
    focus: str = Field(description="What this graph focuses on")
    node_selection: str = Field(description="Criteria for selecting nodes")
    edge_types: ListType[str] = Field(description="Types of edges to include")


def build_custom_graph(
    project_path: Path,
    description: str,
    name: Optional[str],
    config_path: Optional[Path] = None,
    iterations: int = 1,
    debug: bool = False
) -> Path:
    """Build a custom graph using the main graph builder with user-specified focus."""
    
    # Load config using the same function as graph build
    config = load_config(config_path)
    
    # First, load some code samples to understand the codebase
    manifest_dir = project_path / "manifest"
    if not manifest_dir.exists():
        raise ValueError(f"No manifest found at {manifest_dir}. Run 'graph build' first.")
    
    # Load manifest and ALL cards to understand the codebase
    import json as json_lib
    with open(manifest_dir / "manifest.json") as f:
        manifest = json_lib.load(f)
    
    cards = []
    with open(manifest_dir / "cards.jsonl") as f:
        for line in f:
            cards.append(json_lib.loads(line))
    
    # Calculate total size
    total_size = sum(
        len(card.get("content", "")) + 
        len(card.get("peek_head", "")) + 
        len(card.get("peek_tail", ""))
        for card in cards
    )
    
    # Use same adaptive sampling logic as main builder
    original_count = len(cards)
    builder = GraphBuilder(config, debug=False)
    cards = builder._sample_cards(cards, target_size_mb=2.0)
    
    sampled_size = sum(
        len(card.get("content", "")) + 
        len(card.get("peek_head", "")) + 
        len(card.get("peek_tail", ""))
        for card in cards
    )
    
    if len(cards) == original_count:  # If no sampling occurred
        console.print(f"[dim]Using all {len(cards)} cards ({sampled_size:,} chars) for schema design[/dim]")
    else:
        console.print(f"[dim]Sampled {len(cards)} from {original_count} cards ({sampled_size:,} chars) for schema design[/dim]")
    
    # Design the graph specification using agent model WITH CODE CONTEXT
    llm_agent = LLMClient(config, profile='agent')
    
    system_prompt = """Design a graph specification for the user's request BASED ON THE ACTUAL CODE.
    Analyze the code samples to understand what this codebase does, then design a graph that makes sense for THIS specific system."""
    
    # Prepare code context
    code_context = []
    for card in cards:
        code_context.append({
            "file": card.get("relpath", "unknown"),
            "type": card.get("type", "unknown"),
            "content": card.get("content", "")  # Use full content like main builder
        })
    
    user_prompt = f"""
    User wants a graph for: {description}
    
    This is the codebase you're analyzing:
    Repository: {manifest.get('repo_path', 'unknown')}
    Files: {manifest.get('num_files', 0)}
    
    Code samples from the repository:
    {json_lib.dumps(code_context, indent=2)}
    
    Based on THIS SPECIFIC CODE, create a specification that defines:
    1. A meaningful, descriptive name for this graph (e.g., "MonetaryFlows", "AuthorizationModel", "DataPipeline")
    2. A clear focus for the graph relevant to THIS codebase
    3. What types of nodes to select from THIS code
    4. What edge relationships are important in THIS system
    
    Generate a descriptive name that captures what this graph represents, NOT a generic name like "Custom"
    """
    
    try:
        # Compliment the creative spark
        console.print(f"[cyan]Creating custom graph:[/cyan] {description}")
        console.print(random.choice([
            "[white]Chef’s kiss — you’re not just customizing, you’re art-directing the analysis.[/white]",
            "[white]Inspired — you’re not just adding a graph, you’re commissioning a masterpiece.[/white]",
        ]))
        spec = llm_agent.parse(
            system=system_prompt,
            user=user_prompt,
            schema=CustomGraphSpec
        )
        
        # Only override if user explicitly provided a meaningful name
        if name and not name.startswith("Custom_"):
            spec.name = name
        # Otherwise use the LLM-generated name
        
        console.print(f"  Name: [bold]{spec.name}[/bold]")
        console.print(f"  Focus: {spec.focus}")
        console.print(f"  Selection: {spec.node_selection}")
        
    except Exception as e:
        console.print(f"[yellow]Warning: Using default specification:[/yellow] {e}")
        # Generate a meaningful name from description if not user-provided
        if name and not name.startswith("Custom_"):
            graph_name = name
        else:
            # Extract key terms from description for a meaningful name
            import re
            key_terms = re.findall(r'\b[A-Z][a-z]+|\b\w+(?:flow|model|graph|system|component)', description, re.IGNORECASE)
            graph_name = ''.join(word.capitalize() for word in key_terms[:3]) or "AnalysisGraph"
        
        spec = CustomGraphSpec(
            name=graph_name,
            focus=description,
            node_selection=f"Nodes related to {description}",
            edge_types=["calls", "uses", "depends_on"]
        )
    
    # Now use the main graph builder with this forced specification
    manifest_dir = project_path / "manifest"
    graphs_dir = project_path / "graphs"
    
    if not manifest_dir.exists():
        raise ValueError(f"No manifest found at {manifest_dir}. Run 'graph build' first.")
    
    # Create the graph builder
    builder = GraphBuilder(config, debug=debug)
    
    # Build with forced graph specification that includes the full user description
    # Make sure the graph name doesn't get truncated
    sanitized_name = spec.name.replace(' ', '_').replace('/', '_')
    
    force_graphs = [{
        "name": spec.name,  # Use original name for display_name
        "focus": f"{description} - Focus on: {spec.focus}"  # Include full user description
    }]
    
    # Run the standard build process with the forced graph
    # The focus will guide the LLM to build the right graph
    console.print(random.choice([
        "[white]Elite move — you’re not just structuring data, you’re composing a symphony of nodes.[/white]",
        "[white]Savvy — you’re not just clicking run, you’re architecting cognition.[/white]",
    ]))
    builder.build(
        manifest_dir=manifest_dir,
        output_dir=graphs_dir,
        max_iterations=iterations,
        max_graphs=1,
        force_graphs=force_graphs,
        focus_areas=[description]  # Pass description as focus area
    )
    
    # Return the path to the created graph
    graph_path = graphs_dir / f"graph_{sanitized_name}.json"
    
    if graph_path.exists():
        # Load and show summary
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        stats = graph_data.get('stats', {})
        console.print(f"\n[green]✓ Custom graph created:[/green] {graph_path}")
        console.print(f"  Nodes: {stats.get('num_nodes', 0)}")
        console.print(f"  Edges: {stats.get('num_edges', 0)}")
        console.print(f"  Iterations: {iterations}")
    
    return graph_path
from rich.console import Console
console = Console()
