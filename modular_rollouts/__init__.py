from modular_rollouts.vectorized_env import OOP_VecEnv


def create_env(
    env_name,
    action_type,
    n_train_env=1,
    n_eval_env=1,
    env_engine="gym",
    seed=0,
    max_step=None,
    n_pop=1,
    device="cuda",
):
    env = OOP_VecEnv(
        n_env=n_train_env, n_pop=n_pop, max_steps=max_step, seed=seed, device=device
    )
    eval_env = OOP_VecEnv(
        n_env=n_eval_env, n_pop=n_pop, max_steps=max_step, seed=seed, device=device
    )
    env.set_env(
        env_engine=env_engine,
        env_name=env_name,
        action_type=action_type,
    )
    eval_env.set_env(
        env_engine=env_engine,
        env_name=env_name,
        action_type=action_type,
    )
    return env, eval_env
