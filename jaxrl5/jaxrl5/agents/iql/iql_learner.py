"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import math
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
import flax

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal, Normal
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, subsample_ensemble


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def get_weight_decay_mask(params):
     flattened_params = flax.traverse_util.flatten_dict(
         flax.core.frozen_dict.unfreeze(params))

     def decay(k, v):
         if any([(key == 'bias') for key in k]):
             return False
         else:
             return True

     return flax.core.frozen_dict.freeze(
         flax.traverse_util.unflatten_dict(
             {k: decay(k, v)
              for k, v in flattened_params.items()}))


class IQLLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    target_value: TrainState
    discount: float
    tau: float
    expectile: float
    temperature: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: int = struct.field(pytree_node=False)
    num_vs: int = struct.field(pytree_node=False)
    num_min_vs: int = struct.field(pytree_node=False)
    sharing_callback: Optional[Callable[[Any, Any], Any]] = struct.field(pytree_node=False)

    @classmethod
    def create(cls,
               seed: int,
               observation_space: gym.Space,
               action_space: gym.Space,
               actor_lr: Union[float, optax.Schedule] = 1e-3,
               critic_lr: float = 3e-4,
               value_lr: float = 3e-4,
               hidden_dims: Sequence[int] = (256, 256),
               discount: float = 0.99,
               tau: float = 0.005,
               expectile: float = 0.8,
               temperature: float = 0.1,
               actor_weight_decay: Optional[float] = None,
               critic_weight_decay: Optional[float] = None,
               value_weight_decay: Optional[float] = None,
               critic_layer_norm: bool = False,
               value_layer_norm: bool = False,
               num_qs: int = 2,
               num_min_qs: int = None,
               num_vs: int = 1,
               num_min_vs: int = None):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        actions = action_space.sample()
        action_dim = action_space.shape[0]
        actor_base_cls = partial(MLP,
                                 hidden_dims=hidden_dims,
                                 activate_final=True)

        actor_def = Normal(actor_base_cls,
                           action_dim,
                           log_std_min=math.log(0.1),
                           log_std_max=math.log(0.1),
                           state_dependent_std=False)

        observations = observation_space.sample()
        actor_params = actor_def.init(actor_key, observations)['params']

        #if decay_steps is not None:
            #actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_weight_decay is None:
            actor_optimiser = optax.adam(learning_rate=actor_lr)
        else:
            actor_optimiser = optax.adamw(learning_rate=actor_lr,
                                          weight_decay=actor_weight_decay,
                                          mask=get_weight_decay_mask)
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=actor_optimiser)

        critic_base_cls = partial(MLP,
                                  hidden_dims=hidden_dims,
                                  activate_final=True,
                                  use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations,
                                        actions)['params']
        if critic_weight_decay is None:
            critic_optimiser = optax.adam(learning_rate=critic_lr)
        else:
            critic_optimiser = optax.adamw(learning_rate=critic_lr,
                                           weight_decay=critic_weight_decay,
                                           mask=get_weight_decay_mask)
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=critic_optimiser)

        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(apply_fn=target_critic_def.apply,
                                          params=critic_params,
                                          tx=optax.GradientTransformation(
                                              lambda _: None, lambda _: None))

        value_base_cls = partial(MLP,
                                 hidden_dims=hidden_dims,
                                 activate_final=True,
                                 use_layer_norm=value_layer_norm)
        value_cls = partial(StateValue, base_cls=value_base_cls)
        value_def = Ensemble(value_cls, num=num_vs)
        value_params = value_def.init(value_key, observations)['params']

        if value_weight_decay is None:
            value_optimiser = optax.adam(learning_rate=value_lr)
        else:
            value_optimiser = optax.adamw(learning_rate=value_lr,
                                          weight_decay=value_weight_decay,
                                          mask=get_weight_decay_mask)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)

        target_value_def = Ensemble(value_cls, num=num_min_vs or num_vs)
        target_value = TrainState.create(apply_fn=target_value_def.apply,
                                          params=value_params,
                                          tx=optax.GradientTransformation(
                                              lambda _: None, lambda _: None))


        return cls(actor=actor,
                   critic=critic,
                   target_critic=target_critic,
                   value=value,
                   target_value=target_value,
                   tau=tau,
                   discount=discount,
                   expectile=expectile,
                   temperature=temperature,
                   rng=rng,
                   num_qs=num_qs,
                   num_min_qs=num_min_qs,
                   num_vs=num_vs,
                   num_min_vs=num_min_vs,
                   sharing_callback=None)

    def update_v(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = self.rng

        # REDQ-style ensembling, if enabled. Not necessary for purely offline
        # training but helpful for parity with online training
        key, rng = jax.random.split(rng)
        target_critic_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        qs = self.target_critic.apply_fn(
            {'params': target_critic_params}, batch['observations'],
            batch['actions'])
        q = qs.min(axis=0)

        def value_loss_fn(
                value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = self.value.apply_fn({'params': value_params},
                                     batch['observations'])
            value_loss = loss(q[jnp.newaxis] - v, self.expectile).mean()
            return value_loss, {'value_loss': value_loss, 'v': v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(self.value.params)
        value = self.value.apply_gradients(grads=grads)
        target_value = self.target_value.replace(params=self.value.params)

        return self.replace(value=value, target_value=target_value, rng=rng), info

    def update_q(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = self.rng

        key, rng = jax.random.split(rng)
        target_value_params = subsample_ensemble(
            key, self.target_value.params, self.num_min_vs, self.num_vs
        )

        next_vs = self.value.apply_fn({'params': target_value_params},
                                      batch['next_observations'])
        # TODO: Why even ensemble v?
        next_v = next_vs.min(axis=0)

        target_q = batch['rewards'] + self.discount * batch['masks'] * next_v

        def critic_loss_fn(
                critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn({'params': critic_params},
                                           batch['observations'],
                                           batch['actions'])
            critic_loss = ((qs - target_q[jnp.newaxis]) ** 2).sum(axis=0).mean()
            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': jnp.mean(qs),
                'q_std': jnp.std(qs, axis=0).mean()
            }

        grads, info = jax.grad(critic_loss_fn,
                               has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            self.critic.params,
            self.target_critic.params,
            self.tau
        )
        target_critic = self.target_critic.replace(
            params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic), info

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        vs = self.value.apply_fn({'params': self.value.params},
                                 batch['observations'])
        v = vs.mean(axis=0)

        qs = self.target_critic.apply_fn(
            {'params': self.target_critic.params}, batch['observations'],
            batch['actions'])
        q = qs.min(axis=0)

        exp_a = jnp.exp((q - v) * self.temperature)
        exp_a = jnp.minimum(exp_a, 100.0)
        def actor_loss_fn(
                actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({'params': actor_params},
                                       batch['observations'],
                                       training=True)
            
            log_probs = dist.log_prob(batch['actions'])
            actor_loss = -(exp_a * log_probs).mean()

            return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor), info

    @partial(jax.jit)
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, value_info = new_agent.update_v(batch)
        new_agent, actor_info = new_agent.update_actor(batch)

        if self.sharing_callback:
            new_agent = new_agent.replace(
                critic=self.sharing_callback(
                    source=new_agent.value,
                    target=new_agent.critic,
                ),
                target_critic=self.sharing_callback(
                    source=new_agent.target_value,
                    target=new_agent.target_critic,
                ),
            )
        new_agent, critic_info = new_agent.update_q(batch)
        if self.sharing_callback:
            new_agent = new_agent.replace(
                value=self.sharing_callback(
                    source=new_agent.critic,
                    target=new_agent.value,
                ),
                target_value=self.sharing_callback(
                    source=new_agent.target_critic,
                    target=new_agent.target_value,
                ),
            )

        return new_agent, {**actor_info, **critic_info, **value_info}
