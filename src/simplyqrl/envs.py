import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

def make_vec_env(base, *, num_envs: int, seed: int):
    """
    Crea un SyncVectorEnv con `num_envs` copias del entorno.
    - base puede ser:
        · str → id de Gymnasium ("CartPole-v1")
        · callable → factory que devuelva un gym.Env
        · gym.Env  → se recrea por base.spec.id (sin deepcopy)
    """
    if isinstance(base, str):
        factory = lambda: gym.make(base)
    elif callable(base):
        factory = base
    else:                                   
        # Existing instance
        if base.spec is None or base.spec.id is None:
            raise ValueError("El env pasado debe tener .spec.id")
        factory = lambda: gym.make(base.spec.id)

    def _thunk(idx):
        def _inner():
            env = factory()
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + idx)
            return env
        return _inner

    return SyncVectorEnv([_thunk(i) for i in range(num_envs)])