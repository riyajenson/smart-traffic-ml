from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Optional
from collections import deque
import time


@dataclass
class WindowState:
    window_seconds: float
    _samples: Deque[tuple[float, float]] = field(default_factory=deque)  # (t, speed)

    def add_speed(self, speed_cm_s: float, t: Optional[float] = None) -> None:
        if t is None:
            t = time.time()
        self._samples.append((t, speed_cm_s))
        self._evict_old(t)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def features(self, now: Optional[float] = None) -> dict[str, float]:
        if now is None:
            now = time.time()
        self._evict_old(now)

        n = len(self._samples)
        if n == 0:
            return {"avg_speed": 0.0, "vehicle_count": 0.0, "flow_rate": 0.0}

        avg_speed = sum(s for _, s in self._samples) / n
        flow_rate = n / self.window_seconds
        return {"avg_speed": float(avg_speed), "vehicle_count": float(n), "flow_rate": float(flow_rate)}

