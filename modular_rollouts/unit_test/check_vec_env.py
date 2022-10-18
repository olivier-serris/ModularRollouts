import isaacgym
import torch
import chex
import jax.numpy as jnp

import os
from hydra import initialize

import multiprocessing
from multiprocessing import Process
from modular_rollouts.IsaacGymEnvs.utils import get_isaac_cfg
from modular_rollouts.vectorized_env import OOP_VecEnv


def launch_in_process(function, use_main_process=False, **kwargs):
    # Creating an ISAAC env twice ina row is currently bugged.
    # Launching in a subprocess ensure that memory is cleaned
    # Also usefull to clean memory  occupied by JAX
    if use_main_process:
        function(**kwargs)
    else:
        p = Process(target=function, kwargs=kwargs)
        p.start()
        p.join()
        if p.exitcode != 0:
            raise ValueError(f"a process exited with code {p.exitcode}")


class AllCombination:
    n_pop = 4
    n_env = 2
    max_steps = None
    seed = 0
    hidden_sizes = [10, 10]

    def check_env(
        self,
        env_name,
        env_engine,
        action_type,
        actor_device="cuda",
        **env_kwargs,
    ):
        env = OOP_VecEnv(
            n_pop=self.n_pop,
            n_env=self.n_env,
            max_steps=self.max_steps,
            seed=self.seed,
            device="cuda",
        )
        env.set_env(
            env_engine=env_engine,
            env_name=env_name,
            action_type=action_type,
            **env_kwargs,
        )

        env.reset()
        for _ in range(10000):
            action = env.action_space.sample()  # TODO: check how to handle this
            if issubclass(action_type, chex.Array):
                action = jnp.array(action)
            elif issubclass(action_type, torch.Tensor):
                action = torch.tensor(action)
            env.step(action)
        print(f"SUCCESS {env_name,env_engine,action_type,actor_device}")

    def test_brax_torch(self):
        launch_in_process(
            self.check_env,
            env_name="acrobot",
            env_engine="brax",
            action_type=torch.Tensor,
        )

    def test_brax_jax(self):
        launch_in_process(
            self.check_env,
            env_name="acrobot",
            env_engine="brax",
            action_type=chex.Array,
        )

    def test_gym_torch(self):
        launch_in_process(
            self.check_env,
            env_name="CartPole-v1",
            env_engine="gym",
            action_type=torch.Tensor,
        )

    def test_gym_jax(self):
        launch_in_process(
            self.check_env,
            env_name="CartPole-v1",
            env_engine="gym",
            action_type=chex.Array,
        )

    def test_isaac_torch(self):
        # env_name = "FrankaCabinet"
        env_name = "FrankaCubeStack"
        task_cfg = get_isaac_cfg(env_name)
        launch_in_process(
            self.check_env,
            env_name=env_name,
            env_engine="isaac",
            action_type=torch.Tensor,
            cfg=task_cfg,
            headless=False,
            force_render=True,
        )

    def test_isaac_jax(self):
        task_cfg = get_isaac_cfg("Cartpole")
        launch_in_process(
            self.check_env,
            env_name="Cartpole",
            env_engine="isaac",
            action_type=chex.Array,
            cfg=task_cfg,
        )


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    multiprocessing.set_start_method("spawn")
    initialize(config_path="configs/", job_name="get_isaac_cfg")
    debug = AllCombination()
    # debug.test_gym_torch()
    # debug.test_gym_jax()
    # debug.test_brax_torch()
    # debug.test_brax_jax()
    debug.test_isaac_torch()
    debug.test_isaac_jax()
