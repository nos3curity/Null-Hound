"""
Autonomous agent system for security analysis.
"""

from .agent_core import AutonomousAgent

# Export aliases for compatibility
Agent = AutonomousAgent
SimpleAgent = AutonomousAgent  # Backward compatibility

__all__ = ['AutonomousAgent', 'Agent', 'SimpleAgent']