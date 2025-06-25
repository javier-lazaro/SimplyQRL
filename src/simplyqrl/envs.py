import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

def make_vec_env(base, *, num_envs: int, seed: int):
    """
    Creates a `SyncVectorEnv` with `num_envs` parallel copies of a given Gymnasium
    environment. Supports passing an environment ID, a factory, or an existing instance.

    Args:
        base (str | Callable | gym.Env): environment source. Can be:
            - str: Gymnasium ID (e.g. "CartPole-v1")
            - callable: factory returning a new gym.Env
            - gym.Env: existing instance (copied via env.spec.id)
        num_envs (int): number of parallel copies to create
        seed (int): base seed; each copy receives *seed + i*

    Returns:
        SyncVectorEnv: vectorised environment with episode stats automatically recorded

    Raises:
        ValueError: if `base` is a gym.Env without a valid `spec.id`


    """
    if isinstance(base, str):
        factory = lambda: gym.make(base)
    elif callable(base):
        factory = base
    else:                                   
        # Existing instance
        if base.spec is None or base.spec.id is None:
            raise ValueError("The provided gym.Env must have a valid spec.id.")
        factory = lambda: gym.make(base.spec.id)

    def _thunk(idx):
        def _inner():
            env = factory()
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + idx)
            return env
        return _inner

    return SyncVectorEnv([_thunk(i) for i in range(num_envs)])