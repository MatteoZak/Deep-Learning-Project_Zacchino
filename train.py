import argparse
import yaml
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.utils import get_linear_fn
from envs.parametric_lunarlander import ParametricLunarLander
from schedulers.threshold import ThresholdScheduler
from schedulers.retreating_threshold import RetreatingThresholdScheduler
from schedulers.entropy import EntropyScheduler
from schedulers.td_error import TDErrorScheduler
from schedulers.success_rate import SuccessRateScheduler
from schedulers.continuous import ContinuousScheduler
from callbacks.curriculum_callback import CurriculumCallback
from callbacks.best_model_callback import BestModelCallback


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_scheduler(scheduler_type: str, cfg: dict, phases: list[dict]):
    if scheduler_type == "threshold":
        return ThresholdScheduler(
            phases=phases,
            reward_thresholds=cfg["scheduler"]["threshold"]["reward_thresholds"],
            window=cfg["scheduler"]["threshold"]["window"],
        )
    elif scheduler_type == "threshold_retreat":
        return RetreatingThresholdScheduler(
            phases=phases,
            reward_thresholds=cfg["scheduler"]["threshold_retreat"]["reward_thresholds"],
            window=cfg["scheduler"]["threshold_retreat"]["window"],
            retreat_window=cfg["scheduler"]["threshold_retreat"]["retreat_window"],
            retreat_ratio=cfg["scheduler"]["threshold_retreat"]["retreat_ratio"],
        )
    elif scheduler_type == "entropy":
        return EntropyScheduler(
            phases=phases,
            entropy_thresholds=cfg["scheduler"]["entropy"]["entropy_thresholds"],
            window=cfg["scheduler"]["entropy"]["window"],
        )
    elif scheduler_type == "td_error":
        return TDErrorScheduler(
            phases=phases,
            td_error_thresholds=cfg["scheduler"]["td_error"]["td_error_thresholds"],
            window=cfg["scheduler"]["td_error"]["window"],
        )
    elif scheduler_type == "success_rate":
        return SuccessRateScheduler(
            phases=phases,
            advance_thresholds=cfg["scheduler"]["success_rate"]["advance_thresholds"],
            retreat_thresholds=cfg["scheduler"]["success_rate"]["retreat_thresholds"],
            window=cfg["scheduler"]["success_rate"]["window"],
            retreat_window=cfg["scheduler"]["success_rate"]["retreat_window"],
            ema_alpha=cfg["scheduler"]["success_rate"]["ema_alpha"],
            success_threshold=cfg["scheduler"]["success_rate"]["success_threshold"],
        )
    elif scheduler_type == "success_rate_v1":
        return SuccessRateScheduler(
            phases=phases,
            advance_thresholds=cfg["scheduler"]["success_rate_v1"]["advance_thresholds"],
            retreat_thresholds=cfg["scheduler"]["success_rate_v1"]["retreat_thresholds"],
            window=cfg["scheduler"]["success_rate_v1"]["window"],
            retreat_window=cfg["scheduler"]["success_rate_v1"]["retreat_window"],
            ema_alpha=cfg["scheduler"]["success_rate_v1"]["ema_alpha"],
            success_threshold=cfg["scheduler"]["success_rate_v1"]["success_threshold"],
        )
    elif scheduler_type == "success_rate_v2":
        return SuccessRateScheduler(
            phases=phases,
            advance_thresholds=cfg["scheduler"]["success_rate_v2"]["advance_thresholds"],
            retreat_thresholds=cfg["scheduler"]["success_rate_v2"]["retreat_thresholds"],
            window=cfg["scheduler"]["success_rate_v2"]["window"],
            retreat_window=cfg["scheduler"]["success_rate_v2"]["retreat_window"],
            ema_alpha=cfg["scheduler"]["success_rate_v2"]["ema_alpha"],
            success_threshold=cfg["scheduler"]["success_rate_v2"]["success_threshold"],
        )
    elif scheduler_type == "continuous":
        return ContinuousScheduler(
            phases=phases,
            reward_min=cfg["scheduler"]["continuous"]["reward_min"],
            reward_max=cfg["scheduler"]["continuous"]["reward_max"],
            ema_alpha=cfg["scheduler"]["continuous"]["ema_alpha"],
        )
    elif scheduler_type in ("pcer", "pcer_high", "pcer_low", "pcer_flat", "pcer_flat_low", "pcer_3phase", "pcer_flat_low_3phase"):
        sched_cfg = cfg["scheduler"][scheduler_type]
        return ThresholdScheduler(
            phases=phases,
            reward_thresholds=sched_cfg["reward_thresholds"],
            window=sched_cfg["window"],
        )
    elif scheduler_type == "threshold_3phase":
        return ThresholdScheduler(
            phases=phases,
            reward_thresholds=cfg["scheduler"]["threshold_3phase"]["reward_thresholds"],
            window=cfg["scheduler"]["threshold_3phase"]["window"],
        )
    elif scheduler_type == "threshold_retreat_3phase":
        return RetreatingThresholdScheduler(
            phases=phases,
            reward_thresholds=cfg["scheduler"]["threshold_retreat_3phase"]["reward_thresholds"],
            window=cfg["scheduler"]["threshold_retreat_3phase"]["window"],
            retreat_window=cfg["scheduler"]["threshold_retreat_3phase"]["retreat_window"],
            retreat_ratio=cfg["scheduler"]["threshold_retreat_3phase"]["retreat_ratio"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train(scheduler_type: str, seed: int, cfg: dict, log_dir: str):
    phases = cfg["env"]["phases"]
    ppo_cfg = cfg["ppo"]

    parametric_env = ParametricLunarLander(phases[0])
    monitored_env = Monitor(parametric_env, filename=f"{log_dir}/{scheduler_type}_seed{seed}")

    scheduler = make_scheduler(scheduler_type, cfg, phases)
    phase_log_path = f"{log_dir}/{scheduler_type}_seed{seed}_phases.csv"
    pcer_types = ("pcer", "pcer_high", "pcer_low", "pcer_flat", "pcer_flat_low", "pcer_3phase", "pcer_flat_low_3phase")
    ent_coef_schedule = cfg["scheduler"][scheduler_type]["ent_coef_schedule"] if scheduler_type in pcer_types else None
    curriculum_cb = CurriculumCallback(
        scheduler=scheduler,
        env=parametric_env,
        verbose=1,
        phase_log_path=phase_log_path,
        ent_coef_schedule=ent_coef_schedule,
    )

    total_timesteps = ppo_cfg["total_timesteps"]
    clip_start = ppo_cfg["clip_range"]
    clip_end = ppo_cfg.get("clip_range_final", clip_start / 2)
    clip_range_schedule = get_linear_fn(clip_start, clip_end, 1.0)

    best_path = f"checkpoints/{scheduler_type}_seed{seed}_best"
    best_model_cb = BestModelCallback(save_path=best_path, window=100, verbose=1)

    initial_ent_coef = float(ent_coef_schedule[0]) if ent_coef_schedule is not None else ppo_cfg["ent_coef"]
    model = PPO(
        "MlpPolicy",
        monitored_env,
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        learning_rate=ppo_cfg["learning_rate"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=clip_range_schedule,
        ent_coef=initial_ent_coef,
        device=ppo_cfg["device"],
        tensorboard_log=log_dir,
        seed=seed,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, best_model_cb, ProgressBarCallback()],
        tb_log_name=f"{scheduler_type}_seed{seed}",
        reset_num_timesteps=True,
    )

    save_path = f"checkpoints/{scheduler_type}_seed{seed}_final"
    model.save(save_path)
    print(f"Saved model to {save_path}")
    print(f"Best model saved to {best_path} (peak mean reward: {best_model_cb._best_mean_reward:.1f})")
    monitored_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler",
        choices=["threshold", "threshold_retreat", "entropy", "td_error", "success_rate",
                 "success_rate_v1", "success_rate_v2", "continuous",
                 "pcer", "pcer_high", "pcer_low", "pcer_flat", "pcer_flat_low",
                 "threshold_3phase", "threshold_retreat_3phase", "pcer_3phase", "pcer_flat_low_3phase"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-dir", default="runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args.scheduler, args.seed, cfg, args.log_dir)
