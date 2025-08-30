"""QA module – refactor of finalization naming.

Provides a QA class that wraps the existing Finalizer for backward
compatibility while aligning with the new naming.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .finalization import Finalizer


class QA(Finalizer):
    """Alias of Finalizer using the QA naming."""

    def __init__(
        self,
        graphs_metadata_path: Path,
        manifest_path: Path,
        hypothesis_path: Path,
        agent_id: str,
        config: Optional[Dict] = None,
        debug: bool = False,
    ):
        super().__init__(graphs_metadata_path, manifest_path, hypothesis_path, agent_id, config, debug)

__all__ = ["QA"]

