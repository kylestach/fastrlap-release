"""Implementations of algorithms for continuous control."""

from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple, Type

import gym
import jax
from jaxrl5.agents.agent import Agent
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.drq.augmentations import batched_random_crop
from jaxrl5.agents.iql.iql_learner import IQLLearner
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import MLP, Ensemble, PixelMultiplexer, StateActionValue, StateValue
from jaxrl5.networks.encoders import D4PGEncoder, ResNetV2Encoder


# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][..., :-1]
            next_obs_pixels = batch["observations"][pixel_key][..., 1:]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )

    batch = batch.copy(
        add_or_replace={"observations": obs, "next_observations": next_obs}
    )

    return batch


def _share_encoder(source, target):
    replacers = {}

    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v

    # Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)


class PixelIQLLearner(IQLLearner):
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        encoder: str = "d4pg",
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        num_vs: int = 1,
        num_min_vs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        value_dropout_rate: Optional[float] = None,
        value_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        temperature: float = 1.0,
        backup_entropy: bool = True,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
    ):
        """
        An implementation of IQL with the data regularization from DrQ
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if encoder == "d4pg":
            encoder_cls = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        elif encoder == "resnet":
            encoder_cls = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            stop_gradient=True,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=critic_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        target_critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        target_critic_cls = partial(Ensemble, net_cls=target_critic_cls, num=num_min_qs or num_qs)
        target_critic_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=target_critic_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=value_dropout_rate,
            use_layer_norm=value_layer_norm,
        )
        value_cls = partial(StateValue, base_cls=value_base_cls)
        value_cls = partial(Ensemble, net_cls=value_cls, num=num_vs)
        value_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=value_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        value_params = value_def.init(value_key, observations)["params"]
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.adam(learning_rate=value_lr),
        )

        target_value_cls = partial(StateValue, base_cls=value_base_cls)
        target_value_cls = partial(Ensemble, net_cls=target_value_cls, num=num_min_vs or num_vs)
        target_value_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=target_value_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        target_value = TrainState.create(
            apply_fn=target_value_def.apply,
            params=value_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(key, observations, pixel_key)
                if depth_key is not None:
                    observations = batched_random_crop(key, observations, depth_key)
            return observations

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            value=value,
            target_value=target_value,
            discount=discount,
            tau=tau,
            expectile=expectile,
            temperature=temperature,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            num_vs=num_vs,
            num_min_vs=num_min_vs,
            sharing_callback=_share_encoder,
            data_augmentation_fn=data_augmentation_fn,
        )

    @partial(jax.jit)
    def update(self, batch: DatasetDict):
        new_agent = self

        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)


        rng, key = jax.random.split(new_agent.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        rng, key = jax.random.split(rng)
        next_observations = self.data_augmentation_fn(key, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
            }
        )

        new_agent = new_agent.replace(rng=rng)

        return IQLLearner.update(new_agent, batch)
