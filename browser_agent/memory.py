"""Persistent memory primitives for autonomous browser work."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class MemoryRecord:
    """A single memory item persisted across agent runs."""

    content: str
    kind: str = "observation"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    id: str = field(default_factory=lambda: uuid4().hex)


class MemoryStore:
    """Small JSONL-backed memory store with simple keyword recall."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path else None
        self._records: list[MemoryRecord] = []
        if self.path is not None:
            self._load()

    def remember(self, content: str, *, kind: str = "observation", metadata: dict[str, Any] | None = None) -> MemoryRecord:
        """Persist and return a memory record."""

        record = MemoryRecord(content=content, kind=kind, metadata=metadata or {})
        self._records.append(record)
        self._append(record)
        return record

    def recent(self, limit: int = 10, *, kind: str | None = None) -> list[MemoryRecord]:
        """Return recent memory records, optionally restricted by kind."""

        records = [record for record in self._records if kind is None or record.kind == kind]
        return records[-limit:]

    def search(self, query: str, *, limit: int = 5, kind: str | None = None) -> list[MemoryRecord]:
        """Return records ranked by lightweight token overlap."""

        terms = {term.lower() for term in query.split() if term.strip()}
        if not terms:
            return self.recent(limit, kind=kind)

        scored: list[tuple[int, MemoryRecord]] = []
        for record in self._records:
            if kind is not None and record.kind != kind:
                continue
            haystack = f"{record.kind} {record.content} {json.dumps(record.metadata, sort_keys=True)}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [record for _, record in scored[:limit]]

    def clear(self) -> None:
        """Clear in-memory and on-disk records."""

        self._records.clear()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            self._records.append(MemoryRecord(**json.loads(line)))

    def _append(self, record: MemoryRecord) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
