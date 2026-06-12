import csv
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from schedulers.base import CurriculumScheduler
from envs.parametric_lunarlander import ParametricLunarLander


class CurriculumCallback(BaseCallback):
    """SB3 callback that advances curriculum phase at episode boundaries."""

    def __init__(self, scheduler: CurriculumScheduler, env: ParametricLunarLander, verbose: int = 0, phase_log_path: str | None = None, ent_coef_schedule: list[float] | None = None):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.env = env
        self._total_episodes = 0
        self._last_mean_reward = 0.0
        self._phase_log_path = phase_log_path
        self._phase_log_initialized = False
        self._ent_coef_schedule = ent_coef_schedule

    def _on_training_start(self) -> None:
        if self._ent_coef_schedule is not None:
            self.model.ent_coef = float(self._ent_coef_schedule[0])
            self.logger.record("curriculum/ent_coef", self.model.ent_coef)

    def _apply_ent_coef_schedule(self) -> None:
        if self._ent_coef_schedule is None:
            return
        new_ent_coef = float(self._ent_coef_schedule[self.scheduler.current_phase])
        self.model.ent_coef = new_ent_coef
        self.logger.record("curriculum/ent_coef", new_ent_coef)

    def _log_phase_transition(self, new_phase: int) -> None:
        if self._phase_log_path is None:
            return
        os.makedirs(os.path.dirname(self._phase_log_path) or ".", exist_ok=True)
        write_header = not self._phase_log_initialized
        with open(self._phase_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestep", "phase"])
            if write_header:
                writer.writeheader()
                self._phase_log_initialized = True
            writer.writerow({"timestep": self.num_timesteps, "phase": new_phase})

    def _on_step(self) -> bool:
        ep_info_buffer = self.model.ep_info_buffer
        if not ep_info_buffer:
            return True

        n_episodes = sum(self.locals.get("dones", []))
        if n_episodes == 0:
            return True
        self._total_episodes += n_episodes

        recent_rewards = [ep["r"] for ep in ep_info_buffer]
        mean_reward = float(np.mean(recent_rewards))
        self._last_mean_reward = mean_reward
        entropy = self._get_entropy()

        uses_td_error = getattr(self.scheduler, "uses_td_error", False) is True
        uses_continuous = getattr(self.scheduler, "uses_continuous", False) is True

        if not uses_td_error and not uses_continuous:
            new_episodes = list(ep_info_buffer)[-n_episodes:]
            new_params = None
            for ep in new_episodes:
                result = self.scheduler.step({
                    "mean_reward": mean_reward,
                    "entropy": entropy,
                    "episode_reward": ep["r"],
                })
                if result is not None:
                    new_params = result
            if new_params is not None:
                self.env.set_params(new_params)
                self._apply_ent_coef_schedule()
                self._log_phase_transition(self.scheduler.current_phase)
                if self.verbose > 0:
                    print(f"[Curriculum] Phase changed to {self.scheduler.current_phase} at step {self.num_timesteps}")
                self.logger.record("curriculum/phase", self.scheduler.current_phase)

        if uses_continuous:
            self.scheduler.step({"mean_reward": mean_reward})
            params = self.scheduler.get_current_params({"mean_reward": mean_reward})
            self.env.set_params(params)
            self.logger.record("curriculum/continuous_t", self.scheduler.ema_t)

        self.logger.record("curriculum/mean_reward", mean_reward)
        self.logger.record("curriculum/entropy", entropy)
        return True

    def _on_rollout_end(self) -> None:
        if getattr(self.scheduler, "uses_td_error", False) is not True:
            return
        advantages = self.model.rollout_buffer.advantages
        td_error = float(np.mean(np.abs(advantages)))
        self.logger.record("curriculum/td_error", td_error)
        new_params = self.scheduler.step({"td_error": td_error, "mean_reward": self._last_mean_reward})
        if new_params is not None:
            self.env.set_params(new_params)
            self._log_phase_transition(self.scheduler.current_phase)
            if self.verbose > 0:
                print(f"[Curriculum] Phase advanced to {self.scheduler.current_phase} at step {self.num_timesteps}")
            self.logger.record("curriculum/phase", self.scheduler.current_phase)

    def _get_entropy(self) -> float:
        try:
            return -float(self.model.logger.name_to_value.get("train/entropy_loss", -1.0))
        except Exception:
            return 1.0
