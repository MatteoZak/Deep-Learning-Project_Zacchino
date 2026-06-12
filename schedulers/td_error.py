from collections import deque
from schedulers.base import CurriculumScheduler


class TDErrorScheduler(CurriculumScheduler):
    """Advances phase when mean(|advantages|) drops below a per-phase threshold."""

    uses_td_error: bool = True

    def __init__(self, phases: list[dict], td_error_thresholds: list[float], window: int = 50):
        super().__init__(phases)
        self.td_error_thresholds = td_error_thresholds
        self.window = window
        self._td_error_buffer: deque[float] = deque(maxlen=window)

    def step(self, metrics: dict) -> dict | None:
        self._td_error_buffer.append(metrics["td_error"])

        if self.is_at_last_phase:
            return None
        if len(self._td_error_buffer) < self.window:
            return None

        mean_td_error = sum(self._td_error_buffer) / len(self._td_error_buffer)
        threshold = self.td_error_thresholds[self.current_phase]

        if mean_td_error <= threshold:
            self.current_phase += 1
            self._td_error_buffer.clear()
            return self.phases[self.current_phase]

        return None
