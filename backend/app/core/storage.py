from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JSONRepository(Generic[T]):
    def __init__(
        self,
        path: Path,
        model_cls: type[T],
        default_factory: Callable[[], T],
    ) -> None:
        self.path = path
        self.model_cls = model_cls
        self.default_factory = default_factory
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> T:
        with self._lock:
            if not self.path.exists():
                value = self.default_factory()
                self._write(value)
                return value
            raw = self.path.read_text(encoding="utf-8")
            if not raw.strip():
                value = self.default_factory()
                self._write(value)
                return value
            payload = json.loads(raw)
            return self.model_cls.model_validate(payload)

    def save(self, value: T) -> T:
        with self._lock:
            self._write(value)
            return value

    def _write(self, value: T) -> None:
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(value.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
