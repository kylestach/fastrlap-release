import collections
from typing import Optional, Union, Iterable, Callable

import gym
import gym.spaces
import jax
import numpy as np

from jaxrl5.data.dataset import Dataset, DatasetDict, _sample
from flax.core import frozen_dict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        relabel_fn: Optional[Callable[[DatasetDict], DatasetDict]] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

        self._relabel_fn = relabel_fn

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def sample_future_observation(self, indices: np.ndarray, sample_futures: str):
        if sample_futures == 'uniform':
            ep_begin = indices - _sample(self.dataset_dict['observations']['index'], indices)
            ep_end = ep_begin + _sample(self.dataset_dict['observations']['ep_len'], indices)
            future_indices = np.random.randint(ep_begin, ep_end, indices.shape)
        elif sample_futures == 'exponential':
            ep_len = _sample(self.dataset_dict['observations']['ep_len'], indices)
            indices_in_ep = _sample(self.dataset_dict['observations']['index'], indices)
            ep_begin = indices - indices_in_ep
            ep_end = ep_begin + ep_len
            future_offsets = np.random.exponential(100.0, indices.shape).astype(np.int32) + 1
            offsets_from_ep_begin = (future_offsets + indices - ep_begin) % ep_len
            future_indices = (ep_begin + offsets_from_ep_begin) % self._size
        elif sample_futures == 'exponential_no_wrap':
            future_offsets = np.random.exponential(100.0, indices.shape).astype(np.int32)
            future_indices = (indices + future_offsets + 1) % self._size
        else:
            raise ValueError(f'Unknown sample_futures: {sample_futures}')
        return _sample(self.dataset_dict['observations'], future_indices)

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               sample_futures = None,
               relabel: bool = False) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        samples = super().sample(batch_size, keys, indx)

        if sample_futures:
            samples = frozen_dict.unfreeze(samples)
            samples['future_observations'] = self.sample_future_observation(indx, sample_futures)
            samples = frozen_dict.freeze(samples)

        if relabel and self._relabel_fn is not None:
            samples = frozen_dict.unfreeze(samples)
            samples = self._relabel_fn(samples)
            samples = frozen_dict.freeze(samples)

        return samples
