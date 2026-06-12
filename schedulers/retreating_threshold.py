from collections import deque
from schedulers.base import CurriculumScheduler


class RetreatingThresholdScheduler(CurriculumScheduler):
    """ThresholdScheduler with reward-based retreat.

    Advances when mean reward in advance_buffer >= reward_thresholds[phase].
    Retreats when mean reward in retreat_buffer < reward_thresholds[phase-1] * retreat_ratio.
    Retreat threshold is derived automatically: no extra hyperparameters beyond retreat_ratio.
    """

    def __init__(
        self,
        phases: list[dict],
        reward_thresholds: list[float],
        window: int = 50,
        retreat_window: int = 30,
        retreat_ratio: float = 0.75,
    ):
        super().__init__(phases)
        self.reward_thresholds = reward_thresholds
        self.window = window
        self.retreat_window = retreat_window
        self.retreat_ratio = retreat_ratio
        self._advance_buffer: deque[float] = deque(maxlen=window)
        self._retreat_buffer: deque[float] = deque(maxlen=retreat_window)

    def step(self, metrics: dict) -> dict | None:
        r = metrics["mean_reward"]
        self._advance_buffer.append(r)
        self._retreat_buffer.append(r)

        if not self.is_at_last_phase and len(self._advance_buffer) == self.window:
            mean_adv = sum(self._advance_buffer) / self.window
            if mean_adv >= self.reward_thresholds[self.current_phase]:
                self.current_phase += 1
                self._advance_buffer.clear()
                self._retreat_buffer.clear()
                return self.phases[self.current_phase]

        if self.current_phase > 0 and len(self._retreat_buffer) == self.retreat_window:
            retreat_threshold = self.reward_thresholds[self.current_phase - 1] * self.retreat_ratio
            mean_ret = sum(self._retreat_buffer) / self.retreat_window
            if mean_ret < retreat_threshold:
                self.current_phase -= 1
                self._advance_buffer.clear()
                self._retreat_buffer.clear()
                return self.phases[self.current_phase]

        return None
