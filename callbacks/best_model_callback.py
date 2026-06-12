import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BestModelCallback(BaseCallback):
    """Saves the model whenever mean reward over the last `window` episodes is a new best."""

    def __init__(self, save_path: str, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.window = window
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        ep_info_buffer = self.model.ep_info_buffer
        if len(ep_info_buffer) < self.window:
            return True
        mean_reward = float(np.mean([ep["r"] for ep in ep_info_buffer]))
        if mean_reward > self._best_mean_reward:
            self._best_mean_reward = mean_reward
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"[BestModel] New best: {mean_reward:.1f} → saved to {self.save_path}")
        return True
