# mandatory imports :
from typing import Optional, Union, ClassVar
from typing import Sequence, Union, SupportsIndex
import numpy as np
import gym

# Optional imports :
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

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]


class IsaacToGymWrapper(gym.vector.VectorEnv):
    """
    A wrapper that converts isaac Env to one that follows Gym API.
    !!! INCOMPLETE !!!
    """

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: IsaacEnv, seed: int = 0, backend: Optional[str] = None):
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

    def reset(self):
        return self._env.reset()["obs"]

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs["obs"], reward, done, info

    def unwrapped(self) -> IsaacEnv:
        return self._env


class ReshapeWrapper(gym.Wrapper):
    """
    Reshape vectors input and ouput of the environement.
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

    def observation(self, observation: Union[torch.Tensor, chex.Array]) -> chex.Array:
        return observation.reshape(self.out_reshape)

    def action(self, action: chex.Array) -> torch.Tensor:
        return action.reshape(self.in_reshape)

    def reward(self, reward: torch.Tensor):
        return reward.reshape(self.out_reshape)

    def done(self, done: torch.Tensor):
        return done.reshape(self.out_reshape)

    def info(self, info):
        return info  # see what need to be done

    def reset(self):
        obs = super().reset()
        return self.observation(obs)

    def step(self, action):
        action = self.action(action)
        obs, reward, done, info = super().step(action)
        obs = self.observation(obs)
        reward = self.reward(reward)
        done = self.done(done)
        info = self.info(info)
        return obs, reward, done, info


CONVERT = {
    (chex.Array, np.ndarray): lambda x, device: np.array(
        x
    ),  # TODO verify not necessary
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

    def observation(self, observation: np.ndarray) -> torch.Tensor:
        return self.exit_env(observation, self.device)

    def action(self, action: torch.Tensor) -> np.ndarray:
        return self.enter_env(action, device=self.device)

    def reward(self, reward: np.ndarray):
        return self.exit_env(reward, device=self.device)

    def done(self, done: np.ndarray):
        return self.exit_env(done, device=self.device)

    def info(self, info):
        # TODO : Also need to translate values, depends on the wrapper used and
        # if we have a gym env VS jax env VS brax env (list of dict VS dict of arrays)
        return info

    def reset(self):
        obs = super().reset()
        return self.observation(obs)

    def step(self, action):
        action = self.action(action)
        obs, reward, done, info = super().step(action)
        obs = self.observation(obs)
        reward = self.reward(reward)
        done = self.done(done)
        info = self.info(info)
        return obs, reward, done, info
