from typing import TypeVar
from typing import Union
import multiprocessing
import gym
import torch
import numpy as np
import chex
from brax.envs.wrappers import VectorGymWrapper
from brax import envs as brax_env
from brax.envs import wrappers as brax_wrappers
from gym.vector import VectorEnvWrapper
from modular_rollouts.wrappers import (
    ReshapeWrapper,
    DataConversionWrapper,
    BraxToVectorGymWrapper,
    # IsaacToGymWrapper,
)
from omegaconf import DictConfig

Vector = TypeVar("Vector")


def create_envpool_env(
    env_name: str, n_pop: int, n_env: int, max_steps: int, seed: int, **kwargs
):
    try:
        import envpool
    except ModuleNotFoundError:
        raise Exception(
            "You need to install isaacgym: \https://github.com/sail-sg/envpool"
        )
    num_cpu = multiprocessing.cpu_count()
    num_threads = min(num_cpu, n_env * max_steps)
    env = envpool.make(
        env_name, env_type="gym", num_envs=n_env * n_pop, num_threads=num_threads
    )


def create_brax_env(
    env: Union[str, brax_env.Env],
    n_pop: int,
    n_env: int,
    max_steps: int,
    seed: int,
    action_repeat=1,
    **kwargs,
):
    if isinstance(env, str):
        env = brax_env._envs[env](**kwargs)

    if max_steps is not None:
        env = brax_wrappers.EpisodeWrapper(env, max_steps, action_repeat=action_repeat)
    env = brax_wrappers.AutoResetWrapper(env)
    env = brax_wrappers.VectorWrapper(env, n_env * n_pop)
    env = BraxToVectorGymWrapper(env, seed=seed)
    # env = ReshapeWrapper(
    #     env, in_reshape=(n_pop * n_env, -1), out_reshape=(n_pop, n_env, -1)
    # )
    return env


def create_isaac_env(
    env_name: str,
    n_pop: int,
    n_env: int,
    max_steps: int,
    seed: int,
    cfg: DictConfig,
    device: str,
    headless=True,
    force_render=False,
):
    try:
        from modular_rollouts.IsaacGymEnvs.isaacgymenvs.tasks import (
            isaacgym_task_map,
        )
        from modular_rollouts.wrappers import IsaacToGymWrapper
    except ModuleNotFoundError:
        raise Exception(
            "You need to install isaacgym: \nhttps://developer.nvidia.com/isaac-gym"
        )

    EnvCls = isaacgym_task_map[env_name]
    cfg["seed"] = seed
    cfg["env"]["numEnvs"] = n_pop * n_env
    env = EnvCls(
        cfg=cfg,
        rl_device=device,
        sim_device=device,
        graphics_device_id=0,
        headless=headless,
        virtual_screen_capture=False,
        force_render=force_render,
    )
    env = IsaacToGymWrapper(env, n_pop * n_env)
    # if n_pop * n_env > 1:
    #     env = ReshapeWrapper(
    #         env, in_reshape=(n_pop * n_env, -1), out_reshape=(n_pop, n_env, -1)
    #     )
    return env


def create_gym_env(
    env_name: str,
    n_pop: int,
    n_env: int,
    max_steps: int,
    seed: int,
    render_mode: str = None,
    **kwargs,
):
    env = gym.vector.make(
        env_name,
        num_envs=n_pop * n_env,
        asynchronous=False,
        # autoreset=True,
        # max_episode_steps=max_steps,
        # render_mode=render_mode,
    )
    return env


def add_data_conversion_wrappers(env_data_type, actor_data_type, env, device):
    """Add a wrapper that automaticaly handles vector conversions (Numpy / Pytorch / JAX).
    The action given by the aent is converted to the right data type for the env.
    The output of the env is converted to the right data type for the agent.

    env_data_type : specify the data type used by the environment
    actor_data_type : specify the data type used by the agent
    """
    if env_data_type != actor_data_type:
        env = DataConversionWrapper(env, env_data_type, actor_data_type, device=device)
    return env


def get_obs_type(obs):
    if isinstance(obs, chex.Array):
        return chex.Array
    else:
        return type(obs)


class OOP_VecEnv(VectorEnvWrapper):
    def __init__(
        self,
        n_env: int = 1,
        n_pop: int = 1,
        max_steps: int = None,
        seed: int = 0,
        device="cuda",
    ) -> None:
        self.n_pop = n_pop
        self.n_env = n_env
        self.max_steps = max_steps
        self.seed = seed
        self.device = device

    def set_env(self, env_engine, env_name, action_type, **kwargs):
        assert (
            issubclass(action_type, torch.Tensor)
            or issubclass(action_type, np.ndarray)
            or issubclass(action_type, chex.Array)
        ), f"Unknown action type : {action_type}"
        self.env_engine = env_engine
        self.env_name = env_name
        self.action_type = action_type
        if env_engine == "brax":
            env = create_brax_env(
                env_name, self.n_pop, self.n_env, self.max_steps, self.seed, **kwargs
            )
        elif env_engine == "gym":
            env = create_gym_env(
                env_name, self.n_pop, self.n_env, self.max_steps, self.seed, **kwargs
            )
        elif env_engine == "isaac":
            env = create_isaac_env(
                env_name=env_name,
                n_pop=self.n_pop,
                n_env=self.n_env,
                max_steps=self.max_steps,
                device="cuda",  # make this parametrable
                seed=self.seed,
                **kwargs,
            )
        obs, info = env.reset()
        obs_type = get_obs_type(obs)
        if self.n_pop > 1:
            in_shape = [self.n_pop * self.n_env] + list(env.single_action_space.shape)
            env = ReshapeWrapper(
                env,
                in_reshape=in_shape,
                out_reshape=(self.n_pop, self.n_env, -1),
            )

        env = add_data_conversion_wrappers(obs_type, action_type, env, self.device)
        self.env: Union[VectorGymWrapper, gym.vector.VectorEnv] = env

    def reset(self, seed=None):
        self.last_obs = self.env.reset(seed=seed)
        return self.last_obs

    def step(self, actions):
        return self.env.step(actions)

    # Additional helper properties :
    @property
    def action_dim(self) -> gym.Space:
        assert (
            len(self.env.single_observation_space.shape) == 1
        ), "multidimensional observation not yet supported "
        shape = self.env.single_action_space.shape
        if shape == ():
            return 1
        else:
            return shape[0]

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        if isinstance(self.env, gym.Env):
            return self.env.single_action_space.n
        else:
            return self.env.num_actions

    @property
    def observation_dim(self) -> int:
        # TODO how to handle the case where observation is a dict (goal-conditioned)
        assert (
            len(self.env.single_observation_space.shape) == 1
        ), "multidimensional observation not yet supported "
        shape = self.env.single_observation_space.shape
        if shape == ():
            return 1
        else:
            return shape[0]

    def __del__(self):
        if self.env_engine != "isaac":
            return self.unwrapped.__del__()
