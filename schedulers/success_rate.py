from schedulers.base import CurriculumScheduler


class SuccessRateScheduler(CurriculumScheduler):
    """Advances phase when EMA success rate exceeds advance_thresholds; retreats on drop."""

    def __init__(
        self,
        phases: list[dict],
        advance_thresholds: list[float],
        retreat_thresholds: list[float],
        window: int = 50,
        retreat_window: int = 20,
        ema_alpha: float = 0.05,
        success_threshold: float = 200.0,
    ):
        expected = len(phases) - 1
        if len(advance_thresholds) != expected:
            raise ValueError(
                f"advance_thresholds must have length {expected} (len(phases) - 1), "
                f"got {len(advance_thresholds)}"
            )
        if len(retreat_thresholds) != expected:
            raise ValueError(
                f"retreat_thresholds must have length {expected} (len(phases) - 1), "
                f"got {len(retreat_thresholds)}"
            )
        super().__init__(phases)
        self.advance_thresholds = advance_thresholds
        self.retreat_thresholds = retreat_thresholds
        self.window = window
        self.retreat_window = retreat_window
        self.ema_alpha = ema_alpha
        self.success_threshold = success_threshold
        self.ema_success_rate: float = 0.0
        self._advance_streak: int = 0
        self._retreat_streak: int = 0

    def step(self, metrics: dict) -> dict | None:
        success = 1.0 if metrics["episode_reward"] >= self.success_threshold else 0.0
        self.ema_success_rate = (
            self.ema_alpha * success + (1.0 - self.ema_alpha) * self.ema_success_rate
        )

        if not self.is_at_last_phase:
            if self.ema_success_rate >= self.advance_thresholds[self.current_phase]:
                self._advance_streak += 1
            else:
                self._advance_streak = 0

            if self._advance_streak >= self.window:
                self.current_phase += 1
                self._advance_streak = 0
                self._retreat_streak = 0
                return self.phases[self.current_phase]

        if self.current_phase > 0:
            if self.ema_success_rate < self.retreat_thresholds[self.current_phase - 1]:
                self._retreat_streak += 1
            else:
                self._retreat_streak = 0

            if self._retreat_streak >= self.retreat_window:
                self.current_phase -= 1
                self._advance_streak = 0
                self._retreat_streak = 0
                return self.phases[self.current_phase]

        return None
