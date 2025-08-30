"""Agent compatibility shims.

Refactor introduces Scout (junior) and Strategist (senior) names. This module
preserves existing imports while mapping to the new names.
"""

from .scout import Scout

# Backward-compatible aliases for the junior agent
AutonomousAgent = Scout
Agent = Scout
SimpleAgent = Scout

__all__ = ['AutonomousAgent', 'Agent', 'SimpleAgent', 'Scout']
