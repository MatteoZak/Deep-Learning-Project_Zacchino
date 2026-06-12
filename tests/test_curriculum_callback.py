import pytest
from unittest.mock import MagicMock
from callbacks.curriculum_callback import CurriculumCallback
from schedulers.threshold import ThresholdScheduler

PHASES = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def make_callback(scheduler, dones=None):
    env = MagicMock()
    env.current_phase = 0
    callback = CurriculumCallback(scheduler=scheduler, env=env)
    callback.model = MagicMock()
    callback.model.ep_info_buffer = []
    callback.model.logger = MagicMock()
    callback.num_timesteps = 0
    # Simulate one episode done per step by default
    callback.locals = {"dones": dones if dones is not None else [True]}
    return callback, env


def test_callback_calls_scheduler_on_episode_end():
    scheduler = MagicMock()
    scheduler.step.return_value = None
    scheduler.current_phase = 0
    callback, env = make_callback(scheduler)
    callback.model.ep_info_buffer = [{"r": 120.0, "l": 200}]
    callback._on_step()
    scheduler.step.assert_called_once()


def test_callback_calls_set_params_on_phase_advance():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    callback, env = make_callback(scheduler)
    callback.model.ep_info_buffer = [{"r": 150.0, "l": 200}]
    callback._on_step()
    env.set_params.assert_called_once_with(PHASES[1])


def test_callback_does_not_call_set_params_without_advance():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[200, 300], window=1)
    callback, env = make_callback(scheduler)
    callback.model.ep_info_buffer = [{"r": 100.0, "l": 200}]
    callback._on_step()
    env.set_params.assert_not_called()


def test_callback_fires_beyond_100_episodes():
    scheduler = MagicMock()
    scheduler.step.return_value = None
    scheduler.current_phase = 0
    callback, env = make_callback(scheduler)
    callback.model.ep_info_buffer = [{"r": 120.0, "l": 200}]
    # Simulate 200 _on_step calls — previously would stop after 100
    for _ in range(200):
        callback._on_step()
    assert scheduler.step.call_count == 200


import numpy as np
from schedulers.td_error import TDErrorScheduler


def make_td_callback(scheduler, advantages=None):
    """Make a CurriculumCallback wired with a TDErrorScheduler and mock rollout_buffer."""
    env = MagicMock()
    callback = CurriculumCallback(scheduler=scheduler, env=env)
    callback.model = MagicMock()
    callback.model.ep_info_buffer = []
    callback.model.logger = MagicMock()
    callback.num_timesteps = 0
    callback.locals = {"dones": [False]}
    adv = np.array(advantages if advantages is not None else [0.3, 0.4, 0.5])
    callback.model.rollout_buffer = MagicMock()
    callback.model.rollout_buffer.advantages = adv
    return callback, env


def test_on_rollout_end_calls_scheduler_for_td_error():
    scheduler = TDErrorScheduler(phases=PHASES, td_error_thresholds=[0.8, 0.5], window=1)
    callback, env = make_td_callback(scheduler, advantages=[0.3, 0.4])
    callback._on_rollout_end()
    assert scheduler.current_phase == 1


def test_on_rollout_end_does_not_call_for_threshold_scheduler():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    callback, env = make_td_callback(scheduler, advantages=[0.1])
    callback._on_rollout_end()
    env.set_params.assert_not_called()


def test_on_rollout_end_sets_params_on_phase_advance():
    scheduler = TDErrorScheduler(phases=PHASES, td_error_thresholds=[0.8, 0.5], window=1)
    callback, env = make_td_callback(scheduler, advantages=[0.3])
    callback._on_rollout_end()
    env.set_params.assert_called_once_with(PHASES[1])


def test_last_mean_reward_updated_in_on_step():
    scheduler = TDErrorScheduler(phases=PHASES, td_error_thresholds=[0.8, 0.5], window=50)
    callback, env = make_td_callback(scheduler)
    callback.locals = {"dones": [True]}
    callback.model.ep_info_buffer = [{"r": 175.0, "l": 200}]
    callback._on_step()
    assert callback._last_mean_reward == pytest.approx(175.0)


import numpy as np
from schedulers.success_rate import SuccessRateScheduler
from schedulers.continuous import ContinuousScheduler

PHASES_NEW = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def make_sr_callback(scheduler, episode_reward=250.0):
    env = MagicMock()
    callback = CurriculumCallback(scheduler=scheduler, env=env)
    callback.model = MagicMock()
    callback.model.ep_info_buffer = [{"r": episode_reward, "l": 200}]
    callback.model.logger = MagicMock()
    callback.num_timesteps = 0
    callback.locals = {"dones": [True]}
    return callback, env


def make_cont_callback(scheduler, mean_reward=100.0):
    env = MagicMock()
    callback = CurriculumCallback(scheduler=scheduler, env=env)
    callback.model = MagicMock()
    callback.model.ep_info_buffer = [{"r": mean_reward, "l": 200}]
    callback.model.logger = MagicMock()
    callback.num_timesteps = 0
    callback.locals = {"dones": [True]}
    return callback, env


def test_sr_callback_calls_step_with_episode_reward():
    scheduler = MagicMock()
    scheduler.step.return_value = None
    scheduler.uses_continuous = False
    scheduler.uses_td_error = False
    callback, env = make_sr_callback(scheduler, episode_reward=220.0)
    callback._on_step()
    call_kwargs = scheduler.step.call_args[0][0]
    assert "episode_reward" in call_kwargs
    assert call_kwargs["episode_reward"] == pytest.approx(220.0)


def test_continuous_callback_calls_set_params_every_step():
    scheduler = ContinuousScheduler(
        phases=PHASES_NEW, reward_min=-100.0, reward_max=200.0, ema_alpha=1.0
    )
    callback, env = make_cont_callback(scheduler, mean_reward=50.0)
    callback._on_step()
    env.set_params.assert_called_once()
    params = env.set_params.call_args[0][0]
    assert "gravity" in params
    assert "wind_power" in params
    assert "turbulence_power" in params


def test_continuous_callback_logs_ema_t():
    scheduler = ContinuousScheduler(
        phases=PHASES_NEW, reward_min=0.0, reward_max=200.0, ema_alpha=1.0
    )
    callback, env = make_cont_callback(scheduler, mean_reward=100.0)
    callback._on_step()
    calls = [str(c) for c in callback.model.logger.record.call_args_list]
    assert any("continuous_t" in c for c in calls)


def test_discrete_scheduler_not_affected_by_continuous_block():
    scheduler = ThresholdScheduler(phases=PHASES_NEW, reward_thresholds=[300, 400], window=1)
    callback, env = make_cont_callback(scheduler, mean_reward=50.0)
    callback._on_step()
    env.set_params.assert_not_called()


def test_continuous_callback_updates_ema_t():
    scheduler = ContinuousScheduler(
        phases=PHASES_NEW, reward_min=-100.0, reward_max=200.0, ema_alpha=1.0
    )
    callback, env = make_cont_callback(scheduler, mean_reward=50.0)
    callback._on_step()
    assert scheduler.ema_t == pytest.approx(0.5)


# --- PCER tests ---

def make_pcer_callback(scheduler, episode_reward=150.0, ent_coef_schedule=None):
    env = MagicMock()
    callback = CurriculumCallback(
        scheduler=scheduler, env=env, ent_coef_schedule=ent_coef_schedule
    )
    callback.model = MagicMock()
    callback.model.ent_coef = 0.05
    callback.model.ep_info_buffer = [{"r": episode_reward, "l": 200}]
    callback.model.logger = MagicMock()
    callback.num_timesteps = 0
    callback.locals = {"dones": [True]}
    return callback, env


def test_pcer_sets_initial_ent_coef_on_training_start():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[200, 300], window=1)
    env = MagicMock()
    callback = CurriculumCallback(
        scheduler=scheduler, env=env, ent_coef_schedule=[0.10, 0.05, 0.02]
    )
    callback.model = MagicMock()
    callback.model.ent_coef = 0.05
    callback.model.logger = MagicMock()
    callback._on_training_start()
    assert callback.model.ent_coef == pytest.approx(0.10)


def test_pcer_updates_ent_coef_on_phase_advance():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    callback, env = make_pcer_callback(
        scheduler, episode_reward=150.0, ent_coef_schedule=[0.10, 0.05, 0.02]
    )
    callback.model.ent_coef = 0.10  # phase 0 value
    callback._on_step()
    # phase advanced 0→1, ent_coef should be 0.05
    assert callback.model.ent_coef == pytest.approx(0.05)


def test_pcer_no_ent_coef_change_without_schedule():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    callback, env = make_pcer_callback(scheduler, episode_reward=150.0, ent_coef_schedule=None)
    callback.model.ent_coef = 0.05
    callback._on_step()
    # phase advances but no schedule — ent_coef stays unchanged
    assert callback.model.ent_coef == pytest.approx(0.05)


def test_pcer_no_ent_coef_change_without_phase_advance():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[200, 300], window=1)
    callback, env = make_pcer_callback(
        scheduler, episode_reward=50.0, ent_coef_schedule=[0.10, 0.05, 0.02]
    )
    callback.model.ent_coef = 0.10
    callback._on_step()
    # reward below threshold — no advance — ent_coef unchanged
    assert callback.model.ent_coef == pytest.approx(0.10)
