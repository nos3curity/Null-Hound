"""Concurrent plan store for Strategist sessions.

Atomic, file-locked storage of plan items, similar to HypothesisStore.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from .concurrent_knowledge import ConcurrentFileStore


class PlanStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    DROPPED = "dropped"
    SUPERSEDED = "superseded"


def _make_frame_id(session_id: str, question: str, artifact_refs: List[str]) -> str:
    key = f"{session_id}:{question}:{','.join(sorted(artifact_refs or []))}"
    return f"frame_{hashlib.md5(key.encode()).hexdigest()[:12]}"


@dataclass
class PlanItem:
    frame_id: str
    session_id: str
    question: str
    artifact_refs: List[str] = field(default_factory=list)
    priority: int = 5
    status: str = PlanStatus.PLANNED.value
    rationale: str = ""
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
    investigation_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PlanStore(ConcurrentFileStore):
    """Manage plan frames with concurrent-safe operations."""

    def _get_empty_data(self) -> Dict:
        return {
            "version": "1.0",
            "items": {},
            "metadata": {"total": 0, "last_modified": datetime.now().isoformat()},
        }

    def propose(self, session_id: str, question: str, artifact_refs: Optional[List[str]] = None,
                priority: int = 5, rationale: str = "", created_by: Optional[str] = None) -> Tuple[bool, str]:
        """Add a new plan item with duplicate detection within a session."""
        artifact_refs = artifact_refs or []
        frame_id = _make_frame_id(session_id, question, artifact_refs)

        def update(data):
            items = data["items"]
            if frame_id in items:
                return data, (False, frame_id)
            pi = PlanItem(
                frame_id=frame_id,
                session_id=session_id,
                question=question,
                artifact_refs=artifact_refs,
                priority=priority,
                rationale=rationale,
                created_by=created_by,
            )
            items[frame_id] = asdict(pi)
            data["metadata"]["total"] = len(items)
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            return data, (True, frame_id)

        return self.update_atomic(update)

    def update_status(self, frame_id: str, new_status: PlanStatus, rationale: str = "",
                      investigation_id: Optional[str] = None) -> bool:
        def update(data):
            items = data["items"]
            if frame_id not in items:
                return data, False
            it = items[frame_id]
            it["status"] = new_status.value if isinstance(new_status, PlanStatus) else str(new_status)
            if rationale:
                it["rationale"] = rationale
            if investigation_id:
                it["investigation_id"] = investigation_id
            it["updated_at"] = datetime.now().isoformat()
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            return data, True

        return self.update_atomic(update)

    def list(self, session_id: Optional[str] = None, status: Optional[PlanStatus] = None) -> List[Dict[str, Any]]:
        lock = self._acquire_lock()
        try:
            data = self._load_data()
            items = list(data.get("items", {}).values())
            if session_id:
                items = [x for x in items if x.get("session_id") == session_id]
            if status:
                sval = status.value if isinstance(status, PlanStatus) else str(status)
                items = [x for x in items if x.get("status") == sval]
            # Sort by priority desc then created_at
            items.sort(key=lambda x: (-int(x.get("priority", 0)), x.get("created_at", "")))
            return items
        finally:
            self._release_lock(lock)

    def get(self, frame_id: str) -> Optional[Dict[str, Any]]:
        lock = self._acquire_lock()
        try:
            data = self._load_data()
            return data.get("items", {}).get(frame_id)
        finally:
            self._release_lock(lock)

