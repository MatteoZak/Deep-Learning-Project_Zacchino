from collections import deque
from schedulers.base import CurriculumScheduler


class ThresholdScheduler(CurriculumScheduler):
    """Advances phase when mean episode reward exceeds a per-phase threshold."""

    def __init__(self, phases: list[dict], reward_thresholds: list[float], window: int = 100):
        super().__init__(phases)
        self.reward_thresholds = reward_thresholds
        self.window = window
        self._reward_buffer: deque[float] = deque(maxlen=window)

    def step(self, metrics: dict) -> dict | None:
        self._reward_buffer.append(metrics["mean_reward"])

        if self.is_at_last_phase:
            return None
        if len(self._reward_buffer) < self.window:
            return None

        mean_reward = sum(self._reward_buffer) / len(self._reward_buffer)
        threshold = self.reward_thresholds[self.current_phase]

        if mean_reward >= threshold:
            self.current_phase += 1
            self._reward_buffer.clear()
            return self.phases[self.current_phase]

        return None
