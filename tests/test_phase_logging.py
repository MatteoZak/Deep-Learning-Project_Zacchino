import csv
import os
import tempfile
from unittest.mock import MagicMock
from callbacks.curriculum_callback import CurriculumCallback
from schedulers.threshold import ThresholdScheduler

PHASES = [
    {"gravity": -5.0, "wind_power": 0.0, "turbulence_power": 0.0},
    {"gravity": -8.0, "wind_power": 10.0, "turbulence_power": 0.5},
    {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
]


def make_callback_with_log(scheduler, log_path):
    env = MagicMock()
    env.current_phase = 0
    callback = CurriculumCallback(scheduler=scheduler, env=env, phase_log_path=log_path)
    callback.model = MagicMock()
    callback.model.ep_info_buffer = []
    callback.model.logger = MagicMock()
    callback.num_timesteps = 1000
    callback.locals = {"dones": [True]}
    return callback, env


def test_phase_log_file_created_on_advance():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "phase_log.csv")
        callback, env = make_callback_with_log(scheduler, log_path)
        callback.model.ep_info_buffer = [{"r": 150.0, "l": 200}]
        callback._on_step()
        assert os.path.exists(log_path)


def test_phase_log_has_correct_columns():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "phase_log.csv")
        callback, env = make_callback_with_log(scheduler, log_path)
        callback.model.ep_info_buffer = [{"r": 150.0, "l": 200}]
        callback._on_step()
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0].keys() >= {"timestep", "phase"}


def test_phase_log_records_correct_phase():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "phase_log.csv")
        callback, env = make_callback_with_log(scheduler, log_path)
        callback.model.ep_info_buffer = [{"r": 150.0, "l": 200}]
        callback.num_timesteps = 5000
        callback._on_step()
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert int(rows[0]["phase"]) == 1
        assert int(rows[0]["timestep"]) == 5000


def test_no_log_file_without_path():
    scheduler = ThresholdScheduler(phases=PHASES, reward_thresholds=[100, 200], window=1)
    env = MagicMock()
    callback = CurriculumCallback(scheduler=scheduler, env=env)  # no phase_log_path
    callback.model = MagicMock()
    callback.model.ep_info_buffer = [{"r": 150.0, "l": 200}]
    callback.model.logger = MagicMock()
    callback.num_timesteps = 1000
    callback.locals = {"dones": [True]}
    # Should not raise
    callback._on_step()
