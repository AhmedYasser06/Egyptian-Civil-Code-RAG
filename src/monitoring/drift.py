"""Cosine drift of incoming queries versus a reference set.

Reference = the centroid of the BGE-M3 embeddings of the questions the system was evaluated on
(contract/obligations law). At serving time we keep a sliding window of recent query embeddings and
report

    drift       = 1 - cos(reference centroid, window centroid)
    similarity  = mean cos(query, reference centroid)

Asking a contract-law corpus about family law (marriage, custody, inheritance of personal status)
is a genuine semantic shift: drift rises and similarity falls. ``python -m
src.monitoring.drift_demo`` shows it.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np

from src.monitoring import metrics


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-12, None)


class CosineDriftMonitor:
    def __init__(
        self,
        reference_centroid: np.ndarray,
        baseline_similarity: float = 1.0,
        window: int = 100,
        min_samples: int = 20,
        threshold: float = 0.15,
    ):
        self.centroid = _unit(reference_centroid)
        self.baseline_similarity = float(baseline_similarity)
        self.window: deque[np.ndarray] = deque(maxlen=window)
        self.min_samples = min_samples
        self.threshold = threshold
        self._lock = threading.Lock()

    # ----- reference -----------------------------------------------------------------------
    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray, **kwargs) -> CosineDriftMonitor:
        emb = _unit(embeddings)
        centroid = _unit(emb.mean(axis=0))
        baseline = float((emb @ centroid).mean())
        return cls(centroid, baseline, **kwargs)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, centroid=self.centroid, baseline_similarity=self.baseline_similarity)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> CosineDriftMonitor:
        data = np.load(path)
        return cls(data["centroid"], float(data["baseline_similarity"]), **kwargs)

    # ----- online --------------------------------------------------------------------------
    def observe(self, embedding) -> dict | None:
        """Add one query embedding. Returns the drift stats once the window has enough samples."""
        vec = _unit(np.asarray(embedding))
        with self._lock:
            self.window.append(vec)
            if len(self.window) < self.min_samples:
                return None
            stats = self._stats()
        metrics.DRIFT.set(stats["drift"])
        metrics.SIMILARITY.set(stats["similarity"])
        metrics.DRIFT_ALERT.set(1.0 if stats["alert"] else 0.0)
        return stats

    def _stats(self) -> dict:
        window = np.stack(list(self.window))
        window_centroid = _unit(window.mean(axis=0))
        drift = float(1.0 - window_centroid @ self.centroid)
        similarity = float((window @ self.centroid).mean())
        return {
            "drift": drift,
            "similarity": similarity,
            "similarity_drop": self.baseline_similarity - similarity,
            "n": len(self.window),
            "alert": drift > self.threshold,
        }

    def stats(self) -> dict | None:
        with self._lock:
            return self._stats() if len(self.window) >= self.min_samples else None
