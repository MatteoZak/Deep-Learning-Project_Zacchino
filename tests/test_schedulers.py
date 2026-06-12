import pytest
from schedulers.threshold import ThresholdScheduler

PHASES = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def test_threshold_no_advance_below_threshold():
    s = ThresholdScheduler(phases=PHASES, reward_thresholds=[150, 200], window=3)
    for _ in range(3):
        result = s.step({"mean_reward": 100.0, "entropy": 1.0})
    assert result is None
    assert s.current_phase == 0


def test_threshold_advances_phase_when_above_threshold():
    s = ThresholdScheduler(phases=PHASES, reward_thresholds=[150, 200], window=3)
    for _ in range(3):
        result = s.step({"mean_reward": 160.0, "entropy": 1.0})
    assert result == PHASES[1]
    assert s.current_phase == 1


def test_threshold_does_not_advance_past_last_phase():
    s = ThresholdScheduler(phases=PHASES, reward_thresholds=[150, 200], window=3)
    for _ in range(3):
        s.step({"mean_reward": 160.0, "entropy": 1.0})
    for _ in range(3):
        s.step({"mean_reward": 210.0, "entropy": 1.0})
    assert s.current_phase == 2
    result = s.step({"mean_reward": 300.0, "entropy": 1.0})
    assert result is None
    assert s.current_phase == 2


def test_threshold_window_requires_enough_samples():
    s = ThresholdScheduler(phases=PHASES, reward_thresholds=[150, 200], window=5)
    for _ in range(3):
        result = s.step({"mean_reward": 200.0, "entropy": 1.0})
    assert result is None


from schedulers.entropy import EntropyScheduler

PHASES_E = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def test_entropy_no_advance_above_threshold():
    s = EntropyScheduler(phases=PHASES_E, entropy_thresholds=[0.5, 0.3], window=3)
    for _ in range(3):
        result = s.step({"mean_reward": 100.0, "entropy": 0.8})
    assert result is None
    assert s.current_phase == 0


def test_entropy_advances_phase_when_below_threshold():
    s = EntropyScheduler(phases=PHASES_E, entropy_thresholds=[0.5, 0.3], window=3)
    for _ in range(3):
        result = s.step({"mean_reward": 100.0, "entropy": 0.4})
    assert result == PHASES_E[1]
    assert s.current_phase == 1


def test_entropy_does_not_advance_past_last_phase():
    s = EntropyScheduler(phases=PHASES_E, entropy_thresholds=[0.5, 0.3], window=3)
    for _ in range(3):
        s.step({"mean_reward": 100.0, "entropy": 0.4})
    for _ in range(3):
        s.step({"mean_reward": 100.0, "entropy": 0.2})
    assert s.current_phase == 2
    result = s.step({"mean_reward": 100.0, "entropy": 0.01})
    assert result is None


def test_entropy_window_requires_enough_samples():
    s = EntropyScheduler(phases=PHASES_E, entropy_thresholds=[0.5, 0.3], window=5)
    for _ in range(3):
        result = s.step({"mean_reward": 100.0, "entropy": 0.1})
    assert result is None


from schedulers.td_error import TDErrorScheduler

PHASES_TD = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def test_td_error_no_advance_above_threshold():
    s = TDErrorScheduler(phases=PHASES_TD, td_error_thresholds=[0.8, 0.5], window=3)
    for _ in range(3):
        result = s.step({"td_error": 1.2, "mean_reward": 50.0})
    assert result is None
    assert s.current_phase == 0


def test_td_error_advances_phase_when_below_threshold():
    s = TDErrorScheduler(phases=PHASES_TD, td_error_thresholds=[0.8, 0.5], window=3)
    for _ in range(3):
        result = s.step({"td_error": 0.5, "mean_reward": 50.0})
    assert result == PHASES_TD[1]
    assert s.current_phase == 1


def test_td_error_does_not_advance_past_last_phase():
    s = TDErrorScheduler(phases=PHASES_TD, td_error_thresholds=[0.8, 0.5], window=3)
    for _ in range(3):
        s.step({"td_error": 0.5, "mean_reward": 50.0})
    for _ in range(3):
        s.step({"td_error": 0.3, "mean_reward": 80.0})
    assert s.current_phase == 2
    result = s.step({"td_error": 0.1, "mean_reward": 100.0})
    assert result is None
    assert s.current_phase == 2


def test_td_error_window_requires_enough_samples():
    s = TDErrorScheduler(phases=PHASES_TD, td_error_thresholds=[0.8, 0.5], window=5)
    for _ in range(3):
        result = s.step({"td_error": 0.3, "mean_reward": 50.0})
    assert result is None


def test_td_error_buffer_clears_on_phase_advance():
    s = TDErrorScheduler(phases=PHASES_TD, td_error_thresholds=[0.8, 0.5], window=3)
    for _ in range(3):
        s.step({"td_error": 0.5, "mean_reward": 50.0})
    assert len(s._td_error_buffer) == 0


def test_td_error_uses_td_error_attribute():
    s = TDErrorScheduler(phases=PHASES_TD, td_error_thresholds=[0.8, 0.5], window=3)
    assert getattr(s, "uses_td_error", False) is True


from schedulers.success_rate import SuccessRateScheduler

PHASES_SR = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def _sr(advance_thresholds, retreat_thresholds, window=1, retreat_window=1, ema_alpha=1.0):
    return SuccessRateScheduler(
        phases=PHASES_SR,
        advance_thresholds=advance_thresholds,
        retreat_thresholds=retreat_thresholds,
        window=window,
        retreat_window=retreat_window,
        ema_alpha=ema_alpha,
        success_threshold=200.0,
    )


def test_sr_no_advance_below_threshold():
    s = _sr([0.6, 0.8], [0.2, 0.3])
    result = s.step({"episode_reward": 100.0, "mean_reward": 100.0})
    assert result is None
    assert s.current_phase == 0


def test_sr_advances_when_ema_exceeds_threshold():
    s = _sr([0.6, 0.8], [0.2, 0.3])
    result = s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    assert result == PHASES_SR[1]
    assert s.current_phase == 1


def test_sr_window_requires_consecutive_samples():
    s = _sr([0.6, 0.8], [0.2, 0.3], window=3)
    s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    result = s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    assert result == PHASES_SR[1]
    assert s.current_phase == 1


def test_sr_does_not_advance_past_last_phase():
    s = _sr([0.6, 0.8], [0.2, 0.3])
    s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    assert s.current_phase == 2
    result = s.step({"episode_reward": 300.0, "mean_reward": 300.0})
    assert result is None
    assert s.current_phase == 2


def test_sr_retreats_when_ema_drops():
    s = _sr([0.6, 0.8], [0.2, 0.3])
    # advance to phase 1
    s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    assert s.current_phase == 1
    # drop below retreat threshold
    result = s.step({"episode_reward": 0.0, "mean_reward": 0.0})
    assert result == PHASES_SR[0]
    assert s.current_phase == 0


def test_sr_retreat_window_requires_consecutive_drops():
    s = _sr([0.6, 0.8], [0.2, 0.3], retreat_window=2, ema_alpha=1.0)
    s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    assert s.current_phase == 1
    s.step({"episode_reward": 0.0, "mean_reward": 0.0})
    assert s.current_phase == 1  # not yet retreated
    s.step({"episode_reward": 0.0, "mean_reward": 0.0})
    assert s.current_phase == 0


def test_sr_does_not_retreat_below_phase_0():
    s = _sr([0.6, 0.8], [0.2, 0.3])
    result = s.step({"episode_reward": 0.0, "mean_reward": 0.0})
    assert result is None
    assert s.current_phase == 0


def test_sr_advance_streak_resets_on_failure():
    # window=3: need 3 consecutive successes to advance.
    # 2 successes → 1 failure (streak resets) → 3 more successes → advance on 6th call.
    s = _sr([0.6, 0.8], [0.2, 0.3], window=3)
    # Call 1 & 2: successes, streak reaches 2 (not yet at window=3)
    assert s.step({"episode_reward": 250.0, "mean_reward": 250.0}) is None
    assert s.step({"episode_reward": 250.0, "mean_reward": 250.0}) is None
    assert s.current_phase == 0
    # Call 3: failure, streak resets to 0
    assert s.step({"episode_reward": 0.0, "mean_reward": 0.0}) is None
    assert s.current_phase == 0
    # Calls 4 & 5: successes again, streak at 1 then 2
    assert s.step({"episode_reward": 250.0, "mean_reward": 250.0}) is None
    assert s.step({"episode_reward": 250.0, "mean_reward": 250.0}) is None
    assert s.current_phase == 0
    # Call 6: third consecutive success → advance
    result = s.step({"episode_reward": 250.0, "mean_reward": 250.0})
    assert result == PHASES_SR[1]
    assert s.current_phase == 1


from schedulers.continuous import ContinuousScheduler

PHASES_CONT = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def _cont(reward_min=-100.0, reward_max=200.0, ema_alpha=1.0):
    return ContinuousScheduler(
        phases=PHASES_CONT,
        reward_min=reward_min,
        reward_max=reward_max,
        ema_alpha=ema_alpha,
    )


def test_cont_step_always_returns_none():
    s = _cont()
    assert s.step({"mean_reward": 100.0}) is None
    assert s.step({"mean_reward": 200.0}) is None


def test_cont_get_current_params_at_min_reward():
    s = _cont(reward_min=-100.0, reward_max=200.0)
    s.step({"mean_reward": -100.0})
    params = s.get_current_params({"mean_reward": -100.0})
    assert abs(params["gravity"] - PHASES_CONT[0]["gravity"]) < 1e-6
    assert abs(params["wind_power"] - PHASES_CONT[0]["wind_power"]) < 1e-6
    assert abs(params["turbulence_power"] - PHASES_CONT[0]["turbulence_power"]) < 1e-6


def test_cont_get_current_params_at_max_reward():
    s = _cont(reward_min=-100.0, reward_max=200.0)
    s.step({"mean_reward": 200.0})
    params = s.get_current_params({"mean_reward": 200.0})
    assert abs(params["gravity"] - PHASES_CONT[2]["gravity"]) < 1e-6
    assert abs(params["wind_power"] - PHASES_CONT[2]["wind_power"]) < 1e-6
    assert abs(params["turbulence_power"] - PHASES_CONT[2]["turbulence_power"]) < 1e-6


def test_cont_get_current_params_midpoint():
    s = _cont(reward_min=0.0, reward_max=200.0)
    s.step({"mean_reward": 100.0})
    params = s.get_current_params({"mean_reward": 100.0})
    expected_gravity = PHASES_CONT[0]["gravity"] + 0.5 * (PHASES_CONT[2]["gravity"] - PHASES_CONT[0]["gravity"])
    assert abs(params["gravity"] - expected_gravity) < 1e-6


def test_cont_clamps_below_min():
    s = _cont(reward_min=0.0, reward_max=200.0)
    s.step({"mean_reward": -500.0})
    params = s.get_current_params({"mean_reward": -500.0})
    assert abs(params["gravity"] - PHASES_CONT[0]["gravity"]) < 1e-6


def test_cont_clamps_above_max():
    s = _cont(reward_min=0.0, reward_max=200.0)
    s.step({"mean_reward": 9999.0})
    params = s.get_current_params({"mean_reward": 9999.0})
    assert abs(params["gravity"] - PHASES_CONT[2]["gravity"]) < 1e-6


def test_cont_uses_continuous_attribute():
    s = _cont()
    assert getattr(s, "uses_continuous", False) is True


def test_cont_ema_t_is_public():
    s = _cont(reward_min=0.0, reward_max=200.0)
    s.step({"mean_reward": 100.0})
    assert hasattr(s, "ema_t")
    assert 0.0 <= s.ema_t <= 1.0


from schedulers.retreating_threshold import RetreatingThresholdScheduler

PHASES_RT = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def _rt(thresholds=[50.0, 180.0], window=3, retreat_window=2, retreat_ratio=0.75):
    return RetreatingThresholdScheduler(
        phases=PHASES_RT,
        reward_thresholds=thresholds,
        window=window,
        retreat_window=retreat_window,
        retreat_ratio=retreat_ratio,
    )


def test_rt_no_advance_below_threshold():
    s = _rt()
    for _ in range(3):
        result = s.step({"mean_reward": 40.0})
    assert result is None
    assert s.current_phase == 0


def test_rt_advances_when_above_threshold():
    s = _rt()
    for _ in range(3):
        result = s.step({"mean_reward": 60.0})
    assert result == PHASES_RT[1]
    assert s.current_phase == 1


def test_rt_does_not_advance_past_last_phase():
    s = _rt()
    for _ in range(3):
        s.step({"mean_reward": 60.0})
    for _ in range(3):
        s.step({"mean_reward": 190.0})
    assert s.current_phase == 2
    result = s.step({"mean_reward": 300.0})
    assert result is None
    assert s.current_phase == 2


def test_rt_retreats_after_advance_when_reward_drops():
    s = _rt()
    # advance to phase 1
    for _ in range(3):
        s.step({"mean_reward": 60.0})
    assert s.current_phase == 1
    # drop below retreat threshold (50 * 0.75 = 37.5) for retreat_window steps
    for _ in range(2):
        result = s.step({"mean_reward": 30.0})
    assert result == PHASES_RT[0]
    assert s.current_phase == 0


def test_rt_does_not_retreat_below_phase_0():
    s = _rt()
    assert s.current_phase == 0
    for _ in range(5):
        result = s.step({"mean_reward": -200.0})
    assert result is None
    assert s.current_phase == 0


def test_rt_retreat_streak_resets_on_recovery():
    s = _rt()
    # advance to phase 1
    for _ in range(3):
        s.step({"mean_reward": 60.0})
    assert s.current_phase == 1
    # one drop (retreat_window=2, so one isn't enough)
    s.step({"mean_reward": 30.0})
    assert s.current_phase == 1
    # recovery
    s.step({"mean_reward": 60.0})
    assert s.current_phase == 1
    # two consecutive drops now -> should retreat
    s.step({"mean_reward": 30.0})
    result = s.step({"mean_reward": 30.0})
    assert result == PHASES_RT[0]
    assert s.current_phase == 0


def test_rt_advance_buffer_clears_on_advance():
    s = _rt(window=3)
    for _ in range(3):
        s.step({"mean_reward": 60.0})
    assert s.current_phase == 1
    assert len(s._advance_buffer) == 0


def test_rt_both_buffers_clear_on_retreat():
    s = _rt(window=3, retreat_window=2)
    for _ in range(3):
        s.step({"mean_reward": 60.0})
    assert s.current_phase == 1
    for _ in range(2):
        s.step({"mean_reward": 30.0})
    assert s.current_phase == 0
    assert len(s._advance_buffer) == 0
    assert len(s._retreat_buffer) == 0


def test_rt_retreat_threshold_derived_from_ratio():
    s = _rt(thresholds=[100.0, 200.0], retreat_ratio=0.6)
    # Phase 0 retreat threshold = 100.0 * 0.6 = 60.0
    # Reward of 65.0 should NOT trigger retreat (above 60)
    for _ in range(3):
        s.step({"mean_reward": 110.0})  # advance to phase 1
    assert s.current_phase == 1
    for _ in range(2):
        result = s.step({"mean_reward": 65.0})
    assert result is None  # 65 > 60, no retreat
    assert s.current_phase == 1
    # Reward of 55.0 should trigger retreat (below 60)
    for _ in range(2):
        result = s.step({"mean_reward": 55.0})
    assert result == PHASES_RT[0]
    assert s.current_phase == 0
