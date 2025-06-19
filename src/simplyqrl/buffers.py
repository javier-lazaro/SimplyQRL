import numpy as np
import torch
import gymnasium as gym
from collections import namedtuple

# This matches the fields SB3’s sample() returns
ReplayBufferSamples = namedtuple(
    "ReplayBufferSamples",
    ["observations", "actions", "next_observations", "dones", "rewards"],
)

class ReplayBuffer:
    """
    Minimal replay buffer, compatible with CleanRL’s DQN trainer.
    Stores transitions (obs, next_obs, action, reward, done) and
    returns batches as torch tensors on the given device.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        handle_timeout_termination: bool = False,  # ignored
    ):
        self.device = device
        self.max_size = buffer_size
        self.pos = 0
        self.full = False

        # observation buffers
        obs_shape = observation_space.shape
        self.obs_buf = np.zeros((buffer_size, *obs_shape), dtype=observation_space.dtype)
        self.next_obs_buf = np.zeros((buffer_size, *obs_shape), dtype=observation_space.dtype)

        # action buffer: for Discrete, we store a single int per transition
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = 1
            act_dtype = np.int64
        else:
            # assume Box
            self.action_dim = action_space.shape[0]
            act_dtype = action_space.dtype
        self.act_buf = np.zeros((buffer_size, self.action_dim), dtype=act_dtype)

        # reward & done buffers
        self.rew_buf = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict],
    ) -> None:
        # obs shape: (n_envs, *obs_shape), actions: (n_envs,) or (n_envs,action_dim)
        n_envs = obs.shape[0]
        for idx in range(n_envs):
            self.obs_buf[self.pos] = obs[idx]
            self.next_obs_buf[self.pos] = next_obs[idx]

            a = actions[idx]
            # make sure actions get shape (action_dim,)
            if np.isscalar(a):
                a = np.array([a], dtype=self.act_buf.dtype)
            self.act_buf[self.pos] = a

            self.rew_buf[self.pos] = rewards[idx]
            # treat any done/truncation as terminal
            self.done_buf[self.pos] = float(dones[idx])

            self.pos += 1
            if self.pos >= self.max_size:
                self.full = True
                self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        max_mem = self.max_size if self.full else self.pos
        idxs = np.random.randint(0, max_mem, size=batch_size)

        obs      = torch.as_tensor(self.obs_buf[idxs],      device=self.device)
        acts     = torch.as_tensor(self.act_buf[idxs],      device=self.device)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device)
        rews     = torch.as_tensor(self.rew_buf[idxs],      device=self.device)
        dones    = torch.as_tensor(self.done_buf[idxs],     device=self.device)

        return ReplayBufferSamples(
            observations=obs,
            actions=acts,
            next_observations=next_obs,
            dones=dones,
            rewards=rews,
        )
