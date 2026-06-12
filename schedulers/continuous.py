from schedulers.base import CurriculumScheduler


class ContinuousScheduler(CurriculumScheduler):
    """Interpolates difficulty parameters continuously as a function of EMA reward."""

    uses_continuous: bool = True

    def __init__(
        self,
        phases: list[dict],
        reward_min: float = -100.0,
        reward_max: float = 200.0,
        ema_alpha: float = 0.02,
    ):
        if reward_max <= reward_min:
            raise ValueError(f"reward_max ({reward_max}) must be greater than reward_min ({reward_min})")
        super().__init__(phases)
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.ema_alpha = ema_alpha
        self._ema_reward: float = reward_min
        self.ema_t: float = 0.0

    def step(self, metrics: dict) -> dict | None:
        r = metrics["mean_reward"]
        self._ema_reward = self.ema_alpha * r + (1.0 - self.ema_alpha) * self._ema_reward
        span = self.reward_max - self.reward_min
        self.ema_t = max(0.0, min(1.0, (self._ema_reward - self.reward_min) / span))
        return None

    def get_current_params(self, metrics: dict) -> dict:
        easy = self.phases[0]
        hard = self.phases[2]
        return {
            key: easy[key] + self.ema_t * (hard[key] - easy[key])
            for key in easy
        }
