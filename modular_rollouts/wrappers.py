# mandatory imports :
from distutils.log import info
import sys
from typing import Callable, Dict, Optional, Union, ClassVar
from typing import Sequence, Union, SupportsIndex
import numpy as np
import gym
import torch
import jax

# Optional imports :
# TODO : handle optionnal imports in a reliable way
try:
    import jax.numpy as jnp
    import torch
    import brax.io.torch as brax_torch
    from brax.envs import wrappers
    import chex
    from modular_rollouts.IsaacGymEnvs.isaacgymenvs.tasks.base.vec_task import (
        Env as IsaacEnv,
    )
except:
    pass

# Any array
Array = Union[np.ndarray, torch.Tensor, chex.Array]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

if "isaacgym" in sys.modules:

    class IsaacToGymWrapper(gym.vector.VectorEnv):
        """
        A wrapper that converts isaac Env to one that follows Gym API.
        !!! INCOMPLETE !!!
        """

        # Flag that prevents `gym.register` from misinterpreting the `_step` and
        # `_reset` as signs of a deprecated gym Env API.
        _gym_disable_underscore_compat: ClassVar[bool] = True

        def __init__(
            self,
            env: IsaacEnv,
            num_envs: int = 1,
            seed: int = 0,
            backend: Optional[str] = None,
        ):
            self._env = env
            self.seed(seed)
            self.backend = backend
            self._state = None

            self.single_observation_space = self._env.observation_space
            self.single_action_space = self._env.action_space
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space, self._env.num_environments
            )
            self.action_space = gym.vector.utils.batch_space(
                self.single_action_space, self._env.num_environments
            )
            super().__init__(
                num_envs, self.single_observation_space, self.single_action_space
            )

        def reset(self, seed=None, **kwargs):
            return self._env.reset(**kwargs)["obs"]

        def step(self, action):
            obs, reward, done, info = self._env.step(action)
            return obs["obs"], reward, done, info

        def unwrapped(self) -> IsaacEnv:
            return self._env


class ReshapeWrapper(gym.Wrapper):
    """
    Reshape vectors input and ouput of the environement.
    env --> obs --> reshape(out_size) --> agent(obs) --> action
    env <-- act <-- reshape(in_size)  <-- action <-- agent(obs)
    """

    def __init__(
        self,
        env: Union[wrappers.GymWrapper, wrappers.VectorGymWrapper],
        in_reshape: _ShapeLike,
        out_reshape: _ShapeLike,
    ):
        super().__init__(env)
        self.in_reshape: _ShapeLike = in_reshape
        self.out_reshape: _ShapeLike = out_reshape

    def observation(self, observation: Array) -> Array:
        return observation.reshape(self.out_reshape)

    def action(self, action: Array) -> Array:
        return action.reshape(self.in_reshape)

    def reward(self, reward: Array) -> Array:
        return reward.reshape(self.out_reshape)

    def terminated(self, terminated: Array) -> Array:
        return terminated.reshape(self.out_reshape)

    def truncated(self, truncated: Array) -> Array:
        return truncated.reshape(self.out_reshape)

    def info(self, info: Dict) -> Dict:
        return info  # see what need to be done

    def reset(self, **kwargs):
        obs, infos = super().reset(**kwargs)
        return self.observation(obs), infos

    def step(self, action):
        action = self.action(action)

        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.observation(obs)
        reward = self.reward(reward)
        terminated = self.terminated(terminated)
        truncated = self.truncated(truncated)
        info = self.info(info)
        return obs, reward, terminated, truncated, info


CONVERT = {
    (chex.Array, np.ndarray): lambda x, device: np.array(
        x
    ),  # TODO verify , maybe not necessary
    (chex.Array, torch.Tensor): brax_torch.jax_to_torch,
    (np.ndarray, chex.Array): lambda x, device: jnp.array(x),
    (np.ndarray, torch.Tensor): lambda x, device: torch.tensor(x, device=device),
    (torch.Tensor, chex.Array): lambda x, device: brax_torch.torch_to_jax(x),
    (torch.Tensor, np.ndarray): lambda x, device: x.detach().to("cpu").numpy(),
}


class DataConversionWrapper(gym.Wrapper):
    """
    Convert vectors input and ouput of the environement.
    The conversions can be hand-made if a enter_env / exit_env
    are provided.
    Else one of the ready-made conversions are avaiblables :
    JAX     <->     TORCH
    JAX     <->     NUMPY
    TORCH   <->     NUMPY
    """

    def __init__(
        self,
        env: Union[wrappers.GymWrapper, wrappers.VectorGymWrapper],
        env_data_type,
        action_data_type,
        device="cpu",
        enter_env=None,
        exit_env=None,
    ):
        super().__init__(env)
        if enter_env is None:
            assert (
                action_data_type,
                env_data_type,
            ) in CONVERT, f"unknown pair : ({action_data_type}, {env_data_type}), you need to provide your own enter_env fct"
            enter_env = CONVERT[(action_data_type, env_data_type)]
        if exit_env is None:
            assert (
                env_data_type,
                action_data_type,
            ) in CONVERT, f"unknown pair : ({action_data_type}, {env_data_type}), you need to provide your own exit_env fct"
            exit_env = CONVERT[(env_data_type, action_data_type)]

        self.enter_env = enter_env
        self.exit_env = exit_env
        self.device = device

    def observation(self, observation: Array) -> Array:
        return self.exit_env(observation, self.device)

    def action(self, action: Array) -> Array:
        return self.enter_env(action, device=self.device)

    def reward(self, reward: Array) -> Array:
        return self.exit_env(reward, device=self.device)

    def terminated(self, terminated: Array) -> Array:
        return self.exit_env(terminated, device=self.device)

    def truncated(self, truncated: Array) -> Array:
        return self.exit_env(truncated, device=self.device)

    def info(self, info: Dict) -> Dict:
        # TODO : Also need to translate values, depends on the wrapper used and
        # if we have a gym env VS jax env VS brax env (list of dict VS dict of arrays)
        return info

    def reset(self, **kwargs):
        obs, infos = super().reset(**kwargs)
        return self.observation(obs), infos

    def step(self, action):
        action = self.action(action)
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.observation(obs)
        terminated = self.terminated(terminated)
        truncated = self.truncated(truncated)
        reward = self.reward(reward)
        info = self.info(info)
        return obs, reward, terminated, truncated, info


# The next wrapper is adapted from :
# https://github.com/google/brax/blob/b3e75f9f0a66c19a3d76f232218b5029f116f3dd/brax/envs/wrappers.py#L268
# as it is was not compatible with gym 0.26


from brax.envs import env as brax_env
from gym import spaces
from gym.vector import utils
import brax.jumpy as jp


class BraxToVectorGymWrapper(gym.vector.VectorEnv):
    """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

    # Copyright 2022 The Brax Authors.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: brax_env.Env, seed: int = 0):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.sys.config.dt,
        }
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        self.num_envs = self._env.batch_size
        self.seed(seed)
        self._state = None

        obs_high = jp.inf * jp.ones(self._env.observation_size, dtype="float32")
        self.single_observation_space = spaces.Box(-obs_high, obs_high, dtype="float32")
        self.observation_space = utils.batch_space(
            self.single_observation_space, self.num_envs
        )

        action_high = jp.ones(self._env.action_size, dtype="float32")
        self.single_action_space = spaces.Box(
            -action_high, action_high, dtype="float32"
        )
        self.action_space = utils.batch_space(self.single_action_space, self.num_envs)

        def reset(key):
            key1, key2 = jp.random_split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            truncated = self._state.info.get("truncation", jnp.zeros_like(state.reward))
            terminated = jnp.logical_and(state.done, 1 - truncated)
            return state, state.obs, state.reward, terminated, truncated, info

        self._step = jax.jit(step)

    def reset(self, **kwargs):
        # strange that no info can be passed on during reset
        self._state, obs, self._key = self._reset(self._key)
        return obs, {}

    def step(self, action):
        self._state, obs, reward, terminated, truncated, info = self._step(
            self._state, action
        )
        return obs, reward, terminated, truncated, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode="human"):
        # pylint:disable=g-import-not-at-top
        from brax.io import image

        if mode == "rgb_array":
            sys = self._env.sys
            qp = jp.take(self._state.qp, 0)
            return image.render_array(sys, qp, 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception


class RescaleAction(gym.ActionWrapper):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gym
        >>> env = gym.make('BipedalWalker-v3')
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 1.0, 0.75])
        >>> env = RescaleAction(env, min_action=min_action, max_action=max_action)
        >>> env.action_space
        Box(-0.5, [0.   0.5  1.   0.75], (4,), float32)
        >>> RescaleAction(env, min_action, max_action).action_space == gym.spaces.Box(min_action, max_action)
        True
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray ,chex.Array],
        max_action: Union[float, int, np.ndarray,chex.Array],
        array : Callable
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert (min_action < max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )
        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        assert (action>= self.min_action).all(), (
            action,
            self.min_action,
        )
        assert (action <= self.max_action).all(), (action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action
    