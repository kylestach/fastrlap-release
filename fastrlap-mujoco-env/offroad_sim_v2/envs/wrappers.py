import gym
from gym import spaces
import numpy as np
import collections

class KeysToStates(gym.ObservationWrapper):
    """
    Observation wrapper that makes a flat `states` array from a list of keys.
    The keys are kept in the observation dict.
    """

    def __init__(self, env: gym.Env, keys: list):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
            keys: The keys to flatten into a single array
        """
        super().__init__(env, new_step_api=True)
        self.keys = keys
        total_dim = sum([np.prod(self.env.observation_space[k].shape) for k in keys])
        self.observation_space = spaces.Dict({
            **env.observation_space,
            'states': spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
        })

    def observation(self, observation):
        """Concatenate the relevant keys into a single array called 'states'.
        """
        return {
            **observation,
            'states': np.concatenate([observation[k].reshape(-1) for k in self.keys], axis=0),
        }

class PermuteImage(gym.ObservationWrapper):
    """
    Permute the image from DMC's (N, H, W, C) to (H, W, C, N)
    """

    def __init__(self, env: gym.Env, key: str):
        super().__init__(env, new_step_api=True)
        self.key = key
        image_space = self.env.observation_space[key]
        n, h, w, c = image_space.shape
        self.observation_space = spaces.Dict({
            **env.observation_space,
            key: spaces.Box(
                np.transpose(image_space.low, (1, 2, 3, 0)),
                np.transpose(image_space.high, (1, 2, 3, 0)),
                (h, w, c, n),
                image_space.dtype,
            )
        })

    def observation(self, observation):
        """Concatenate the relevant keys into a single array called 'states'.
        """
        return {
            **observation,
            self.key: np.transpose(observation[self.key], (1, 2, 3, 0)),
        }


class RunningReturnInfo(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._return = 0
        self._running_return = 0
        self._max_run_length = None
        self._last_rewards = collections.deque(maxlen=self._max_run_length)

    def step(self, action):
        step_returns = self.env.step(action)
        if len(self._last_rewards) == self._max_run_length:
            self._running_return -= self._last_rewards.popleft()
        if len(step_returns) == 5:
            observation, reward, terminated, truncated, info = step_returns
            self._running_return += reward
            self._last_rewards.append(reward)
            return observation, reward, terminated, truncated, {**info, 'running_return': self._running_return}
        else:
            observation, reward, done, info = step_returns
            self._running_return += reward
            self._last_rewards.append(reward)
            return observation, reward, done, {**info, 'running_return': self._running_return}

    def reset(self):
        self._running_return = 0
        self._last_rewards = collections.deque(maxlen=self._max_run_length)
        return self.env.reset()


class ReplaceKey(gym.ObservationWrapper):
    """
    Observation wrapper to replace a key in the observation dict.
    """

    def __init__(self, env: gym.Env, key: str, new_key: str):
        super().__init__(env, new_step_api=True)
        self._key = key
        self._new_key = new_key
        self.observation_space = spaces.Dict({
            **{k: v for k, v in env.observation_space.items() if k != key},
            self._new_key: env.observation_space[key],
        })

    def observation(self, observation):
        """Concatenate the relevant keys into a single array called 'states'.
        """
        return {
            **{k: v for k, v in observation.items() if k != self._key},
            self._new_key: observation[self._key],
        }