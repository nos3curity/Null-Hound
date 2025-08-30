"""Session tracker with coverage tracking for audit sessions."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field, asdict
import threading


@dataclass
class SessionCoverage:
    """Track coverage statistics for a session."""
    visited_nodes: Set[str] = field(default_factory=set)
    visited_cards: Set[str] = field(default_factory=set)
    total_nodes: int = 0
    total_cards: int = 0
    
    def add_node(self, node_id: str):
        """Mark a node as visited."""
        self.visited_nodes.add(node_id)
    
    def add_card(self, card_id: str):
        """Mark a card as visited."""
        self.visited_cards.add(card_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coverage statistics."""
        nodes_visited = len(self.visited_nodes)
        cards_visited = len(self.visited_cards)
        
        return {
            'nodes': {
                'visited': nodes_visited,
                'total': self.total_nodes,
                'percent': round(100 * nodes_visited / self.total_nodes, 1) if self.total_nodes > 0 else 0.0
            },
            'cards': {
                'visited': cards_visited,
                'total': self.total_cards,
                'percent': round(100 * cards_visited / self.total_cards, 1) if self.total_cards > 0 else 0.0
            },
            'visited_node_ids': list(self.visited_nodes),
            'visited_card_ids': list(self.visited_cards)
        }


class SessionTracker:
    """Track an audit session including coverage, investigations, and planning."""
    
    def __init__(self, session_dir: Path, session_id: str):
        """Initialize session tracker.
        
        Args:
            session_dir: Directory to store session data
            session_id: Unique session identifier
        """
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.session_file = self.session_dir / f"{session_id}.json"
        self.lock = threading.Lock()
        
        # Initialize or load session data
        self.session_data = self._load_or_init()
        
        # Initialize coverage tracker
        self.coverage = SessionCoverage()
        if 'coverage' in self.session_data:
            cov_data = self.session_data['coverage']
            self.coverage.visited_nodes = set(cov_data.get('visited_node_ids', []))
            self.coverage.visited_cards = set(cov_data.get('visited_card_ids', []))
            self.coverage.total_nodes = cov_data.get('nodes', {}).get('total', 0)
            self.coverage.total_cards = cov_data.get('cards', {}).get('total', 0)
    
    def _load_or_init(self) -> Dict[str, Any]:
        """Load existing session or initialize new one."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Initialize new session
        return {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'status': 'active',
            'models': {},
            'investigations': [],
            'planning_history': [],
            'token_usage': {},
            'coverage': {}
        }
    
    def set_models(self, scout_model: str, strategist_model: str):
        """Set the models being used."""
        self.session_data['models'] = {
            'scout': scout_model,
            'strategist': strategist_model
        }
        self._save()
    
    def initialize_coverage(self, graphs_dir: Path, manifest_dir: Path):
        """Initialize coverage tracking by counting total nodes and cards.
        
        Args:
            graphs_dir: Directory containing graph files
            manifest_dir: Directory containing manifest files
        """
        # Count nodes from graphs
        total_nodes = 0
        if graphs_dir.exists():
            for graph_file in graphs_dir.glob("graph_*.json"):
                try:
                    with open(graph_file, 'r') as f:
                        graph_data = json.load(f)
                        nodes = graph_data.get('nodes', [])
                        total_nodes += len(nodes)
                except Exception:
                    pass
        
        # Count cards from manifest
        total_cards = 0
        manifest_file = manifest_dir / "manifest.json" if manifest_dir.exists() else None
        if manifest_file and manifest_file.exists():
            try:
                with open(manifest_file, 'r') as f:
                    manifest_data = json.load(f)
                    # Try both formats - num_cards or files array
                    if 'num_cards' in manifest_data:
                        total_cards = manifest_data['num_cards']
                    elif 'files' in manifest_data:
                        total_cards = len(manifest_data['files'])
            except Exception:
                pass
        
        self.coverage.total_nodes = total_nodes
        self.coverage.total_cards = total_cards
        self._save()
    
    def track_node_visit(self, node_id: str):
        """Track that a node was visited during investigation."""
        with self.lock:
            self.coverage.add_node(node_id)
            self._save()
    
    def track_card_visit(self, card_path: str):
        """Track that a code card was analyzed."""
        with self.lock:
            self.coverage.add_card(card_path)
            self._save()
    
    def track_nodes_batch(self, node_ids: List[str]):
        """Track multiple nodes visited at once."""
        with self.lock:
            for node_id in node_ids:
                self.coverage.add_node(node_id)
            self._save()
    
    def track_cards_batch(self, card_paths: List[str]):
        """Track multiple cards analyzed at once."""
        with self.lock:
            for card_path in card_paths:
                self.coverage.add_card(card_path)
            self._save()
    
    def add_investigation(self, investigation: Dict[str, Any]):
        """Add an investigation to the session history."""
        with self.lock:
            self.session_data['investigations'].append({
                'timestamp': datetime.now().isoformat(),
                **investigation
            })
            self._save()
    
    def add_planning(self, plan_items: List[Dict[str, Any]]):
        """Add a planning batch to the history."""
        with self.lock:
            self.session_data['planning_history'].append({
                'timestamp': datetime.now().isoformat(),
                'items': plan_items
            })
            self._save()
    
    def update_token_usage(self, tokens: Dict[str, Any]):
        """Update token usage statistics."""
        with self.lock:
            # The token tracker passes a complex structure with total_usage, by_model, and history
            # We'll store the entire structure
            self.session_data['token_usage'] = tokens
            self._save()
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get current coverage statistics."""
        return self.coverage.get_stats()
    
    def finalize(self, status: str = 'completed'):
        """Mark session as finalized."""
        with self.lock:
            self.session_data['status'] = status
            self.session_data['end_time'] = datetime.now().isoformat()
            self.session_data['coverage'] = self.coverage.get_stats()
            self._save()
    
    def _save(self):
        """Save session data to file (call within lock)."""
        try:
            # Include current coverage in saved data
            self.session_data['coverage'] = self.coverage.get_stats()
            
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save session data: {e}")