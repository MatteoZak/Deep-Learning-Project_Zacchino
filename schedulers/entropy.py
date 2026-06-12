from collections import deque
from schedulers.base import CurriculumScheduler


class EntropyScheduler(CurriculumScheduler):
    """Advances phase when mean policy entropy drops below a per-phase threshold."""

    def __init__(self, phases: list[dict], entropy_thresholds: list[float], window: int = 100):
        super().__init__(phases)
        self.entropy_thresholds = entropy_thresholds
        self.window = window
        self._entropy_buffer: deque[float] = deque(maxlen=window)

    def step(self, metrics: dict) -> dict | None:
        self._entropy_buffer.append(metrics["entropy"])

        if self.is_at_last_phase:
            return None
        if len(self._entropy_buffer) < self.window:
            return None

        mean_entropy = sum(self._entropy_buffer) / len(self._entropy_buffer)
        threshold = self.entropy_thresholds[self.current_phase]

        if mean_entropy <= threshold:
            self.current_phase += 1
            self._entropy_buffer.clear()
            return self.phases[self.current_phase]

        return None
