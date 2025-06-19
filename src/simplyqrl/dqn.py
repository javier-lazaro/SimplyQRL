import time
import random
import torch
import copy
import csv
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .agents import build_agent
from .buffers import ReplayBuffer, ReplayBufferSamples


def make_env(env_id, seed, idx, capture_video=False, run_name="experiment"):
    def thunk():
        # If you want to capture a video, you can enable this if needed:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linearly decays epsilon from start_e to end_e over 'duration' steps."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.array(env.single_observation_space.shape).prod()
        act_shape = env.single_action_space.n

        self.network = nn.Sequential(
            nn.Linear(obs_shape, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, act_shape),
        )

    def forward(self, x):
        return self.network(x)

class DQN():
    def __init__(
            self,
            env,
            env_id="CartPole-v1",
            seed=1,
            agent_type="mlp",
            agent_config=None,
            learning_rate=2.5e-4,
            buffer_size=10000,
            gamma=0.99,
            tau=1.0,
            target_network_frequency=500,
            batch_size=128,
            start_e=1.0,
            end_e=0.05,
            exploration_fraction=0.5,
            learning_starts=10000,
            train_frequency=10,
            run_name="DQN",
            capture_video=False,
            verbose=False
        ):


        # Static (hard-coded) hyperparameters and settings
        self.env_id = env_id
        self.seed = seed
        self.run_name = run_name
        self.capture_video = capture_video
        self.verbose = verbose

        # DQN hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.batch_size = batch_size
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency

        # Seed and Torch settings
        self.torch_deterministic = True
        self.cuda = True

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        # Create environment (1 env) - Multiple enviroments are not supported

        env_fns = []

        if env is not None:
            env.reset(seed=seed)
            # Vectorize deep‐copies of the provided env
            for idx in range(1):
                def thunk(base_env=env, idx=idx):
                    env_copy = copy.deepcopy(base_env)
                    env_copy = gym.wrappers.RecordEpisodeStatistics(env_copy)
                    env_copy.reset(seed=seed) #opt + idx
                    return env_copy
                env_fns.append(thunk)

        elif env_id is not None:
            # Use existing make_env helper (assumed to wrap stats & reset seed)
            for idx in range(1):
                env_fns.append(make_env(env_id, seed, idx))

        # Create a SyncVectorEnv from all thunks
        self.envs = gym.vector.SyncVectorEnv(env_fns)

        """
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(self.env_id, self.seed + i, i, self.capture_video, self.run_name) for i in range(1)]
        )
        """

        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "Only discrete action space is supported."

        # DQN networks
        obs_shape = self.envs.single_observation_space.shape
        n_actions = self.envs.single_action_space.n

        self.q_network = build_agent(agent_type, obs_shape, n_actions, config=agent_config, is_qnet=True) 
        self.target_network = build_agent(agent_type, obs_shape, n_actions, config=agent_config, is_qnet=True) 

        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay Buffer
        self.rb = ReplayBuffer(
            self.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

        # Logger
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.start_time = time.time()

        # Setup CSV logging
        os.makedirs("runs", exist_ok=True)
        self.csv_file = open(f"runs/{self.run_name}.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # Write header
        self.csv_writer.writerow(["episode_n", "ep_reward", "global_step"])
        # Initialize episode counter
        self.episode_counter = 0

        # Initialize first observation
        self.obs, _ = self.envs.reset(seed=self.seed)

    def train(self, total_timesteps=10_000, progress_bar=False):
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar

        pbar = tqdm(total=total_timesteps, desc="Training", unit="step") if progress_bar else None

        for global_step in range(self.total_timesteps + 1):
            # Epsilon-greedy
            epsilon = linear_schedule(
                self.start_e,
                self.end_e,
                int(self.exploration_fraction * self.total_timesteps),
                global_step,
            )
            if random.random() < epsilon:
                actions = np.array(
                    [self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]
                )
            else:
                q_values = self.q_network(torch.Tensor(self.obs).to(self.device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # Step in the environment
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # Logging final episode returns (if any)
            if "episode" in infos and "_episode" in infos:
                for i, done in enumerate(infos["_episode"]):
                    if done:
                        ep_return = float(infos["episode"]["r"][i])
                        ep_length = int(infos["episode"]["l"][i])
                    if self.progress_bar == False:
                        print(f"global_step={global_step}, episodic_return={ep_return}")
                    self.writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    self.writer.add_scalar("charts/episodic_length", ep_length, global_step)

                    # CSV logging
                    self.episode_counter += 1
                    self.csv_writer.writerow([self.episode_counter, ep_return, global_step])
                    self.csv_file.flush()  # ensure the file is updated incrementally

            real_next_obs = next_obs.copy()
            # Save to replay buffer
            self.rb.add(self.obs, real_next_obs, actions, rewards, terminations, infos)
            self.obs = next_obs

            # Training step
            if global_step > self.learning_starts and global_step % self.train_frequency == 0:
                data = self.rb.sample(self.batch_size)

                with torch.no_grad():
                    target_max, _ = self.target_network(data.next_observations).max(dim=1)
                    td_target = (
                        data.rewards.flatten()
                        + self.gamma * target_max * (1 - data.dones.flatten())
                    )

                old_val = (
                    self.q_network(data.observations)
                    .gather(1, data.actions)
                    .squeeze()
                )
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/td_loss", loss, global_step)
                    self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - self.start_time))
                    self.writer.add_scalar("charts/SPS", sps, global_step)

                self.optimizer.zero_grad()
                loss.backward()

                # Check gradient norm occasionally
                if self.verbose:
                    if global_step % 1000 == 0:
                        total_grad_norm = 0.0
                        param_count = 0
                        for param in self.q_network.parameters():
                            if param.grad is not None:
                                total_grad_norm += param.grad.data.norm(2).item()
                                param_count += 1
                        avg_grad_norm = total_grad_norm / max(param_count, 1)
                        print(f"Step {global_step}, TD loss: {loss.item():.4f}, Avg grad norm: {avg_grad_norm:.6f}")

                self.optimizer.step()

            # Update target network
            if global_step % self.target_network_frequency == 0:
                for target_param, param in zip(
                    self.target_network.parameters(), self.q_network.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

            # Print Q-values every 1000 steps
            if self.verbose:
                if global_step % 1000 == 0:
                    test_obs = torch.randn(8, 4).to(self.device)
                    with torch.no_grad():
                        test_q = self.q_network(test_obs)
                    print(f"Step {global_step}: Q mean={test_q.mean():.3f}, std={test_q.std():.3f}")

            # Peek at replay buffer states every 2000 steps
            if self.verbose:
                if global_step % 2000 == 0 and global_step > self.learning_starts:
                    debug_data = self.rb.sample(min(self.batch_size, 5))
                    print("Sampled states:", debug_data.observations.cpu().numpy())
                    print("Sampled actions:", debug_data.actions.cpu().numpy())
                    print("Sampled rewards:", debug_data.rewards.cpu().numpy())
                    print("Sampled dones:", debug_data.dones.cpu().numpy())
            
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        self.envs.close()
        self.writer.close()
        print("Training finished!")

    def close(self):
        """Close the environments and writer."""
        self.envs.close()
        self.writer.close()
        self.csv_file.close()
