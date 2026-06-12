import pytest
import numpy as np
from envs.parametric_lunarlander import ParametricLunarLander

PHASE1 = {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0}
PHASE3 = {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5}


def test_creates_with_default_params():
    env = ParametricLunarLander(PHASE1)
    obs, _ = env.reset()
    assert obs.shape == (8,)
    env.close()


def test_step_returns_valid_transition():
    env = ParametricLunarLander(PHASE1)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    env.close()


def test_set_params_changes_gravity():
    env = ParametricLunarLander(PHASE1)
    env.reset()
    env.set_params(PHASE3)
    assert env.current_params["gravity"] == -10.0
    obs, _ = env.reset()
    assert obs.shape == (8,)
    env.close()


def test_set_params_only_at_episode_boundary():
    env = ParametricLunarLander(PHASE1)
    env.reset()
    env.set_params(PHASE3)
    assert env.current_params["gravity"] == -10.0
    env.close()


def test_observation_space_unchanged_after_set_params():
    env = ParametricLunarLander(PHASE1)
    space_before = env.observation_space
    env.set_params(PHASE3)
    env.reset()
    assert env.observation_space == space_before
    env.close()
