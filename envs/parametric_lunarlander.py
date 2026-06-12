import gymnasium as gym


class ParametricLunarLander(gym.Env):
    """LunarLander-v3 wrapper with runtime-configurable difficulty params.

    Params are applied on the next reset() call to avoid mid-episode discontinuities.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, params: dict, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.current_params = dict(params)
        self._pending_params = None
        self._env = self._make_env(self.current_params)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _make_env(self, params: dict) -> gym.Env:
        return gym.make(
            "LunarLander-v3",
            gravity=params["gravity"],
            wind_power=params["wind_power"],
            turbulence_power=params["turbulence_power"],
            render_mode=self.render_mode,
        )

    def set_params(self, params: dict) -> None:
        """Schedule a difficulty update to take effect on the next reset()."""
        self.current_params = dict(params)
        self._pending_params = dict(params)

    def reset(self, *, seed=None, options=None):
        if self._pending_params is not None:
            self._env.close()
            self._env = self._make_env(self._pending_params)
            self._pending_params = None
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
