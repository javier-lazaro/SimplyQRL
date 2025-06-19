import os, csv, copy, time, math, json, zipfile, tempfile, shutil
import cloudpickle

import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim

from io import BytesIO
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

from .agents import build_agent
from .envs import make_vec_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal layer initialization as used in CleanRL."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_id, seed, idx, capture_video=False, run_name="test"):
    """Create a single sub-environment, with optional video recording."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return thunk


class PPO:
    """
    A standalone version of the CleanRL PPO for discrete action spaces.
    This class sets up multiple parallel environments, trains an MLP actor-critic
    (as in CleanRL), and logs to TensorBoard.
    """

    def __init__(
        self,
        env_id="CartPole-v1",
        env=None,
        seed=1,
        agent_type="mlp",
        agent_config=None,
        learning_rate=2.5e-4,
        num_envs=1,
        num_steps=512,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        run_name="PPO",
        capture_video=False,
    ):
        #self.env_id = env_id
        self.env = env
        self.seed = seed
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.agent_type   = agent_type
        self.agent_config = agent_config
        self.run_name = run_name
        #self.capture_video = capture_video

        # Derived values
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True


        self.envs = make_vec_env(
            base = env if env is not None else env_id,
            num_envs = num_envs,
            seed = seed,
        )

        """
        env_fns = []

        if env is not None:
            env.reset(seed=seed)
            # Vectorize deep‐copies of the provided env
            for idx in range(num_envs):
                def thunk(base_env=env, idx=idx):
                    env_copy = copy.deepcopy(base_env)
                    env_copy = gym.wrappers.RecordEpisodeStatistics(env_copy)
                    env_copy.reset(seed=seed) #opt + idx
                    return env_copy
                env_fns.append(thunk)

        elif env_id is not None:
            # Use existing make_env helper (assumed to wrap stats & reset seed)
            for idx in range(num_envs):
                env_fns.append(make_env(env_id, seed, idx))

        # Create a SyncVectorEnv from all thunks
        self.envs = gym.vector.SyncVectorEnv(env_fns)
        """

        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action spaces supported"

        # Create actor-critic
        obs_shape = self.envs.single_observation_space.shape
        obs_dim   = int(np.prod(obs_shape))
        n_actions = self.envs.single_action_space.n
                  
        # We create the corresponding agent
        self.agent = build_agent(agent_type, obs_shape, n_actions, is_qnet=False, config=agent_config)
        self.agent.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # Create buffers for rollouts
        #self.obs = torch.zeros((self.num_steps, self.num_envs) + obs_shape).to(self.device)
        #self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # Create buffers for rollouts (flattened obs)
        self.obs    = torch.zeros(self.num_steps, self.num_envs, obs_dim).to(self.device)
        self.actions= torch.zeros(self.num_steps, self.num_envs).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # TensorBoard writer
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.global_step = 0

        # Setup CSV logging
        os.makedirs("runs", exist_ok=True)
        self.csv_file = open(f"runs/{self.run_name}.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # Write header
        self.csv_writer.writerow(["episode_n", "ep_reward", "global_step"])
        # Initialize episode counter
        self.episode_counter = 0

    @property
    def device(self):
        return torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, total_timesteps: int = 100_000, progress_bar: bool = False):
        """Main PPO training loop."""
        # Reset envs
        next_obs_np, _ = self.envs.reset(seed=self.seed)
        batch = next_obs_np.reshape(self.num_envs, -1)                                  
        next_obs = torch.tensor(batch, dtype=torch.float32, device=self.device)
        #next_obs, _ = self.envs.reset(seed=self.seed)
        #next_obs = torch.Tensor(next_obs).to(self.device)  
        next_done = torch.zeros(self.num_envs).to(self.device)
        start_time = time.time()

        self.num_iterations = math.ceil(total_timesteps / self.batch_size) 

        pbar = tqdm(total=total_timesteps, desc="Training", unit="step") if progress_bar else None

        for iteration in range(1, self.num_iterations + 1):
            # Annneal learning rate
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Collect experience
            for step in range(self.num_steps):
                self.global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.actions[step] = action
                self.logprobs[step] = logprob
                self.values[step] = value.flatten()

                next_obs_np, reward, terminated, truncated, infos = self.envs.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminated, truncated)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)

                # flatten into (num_envs, feature_dim) before tensor
                batch = next_obs_np.reshape(self.num_envs, -1)
                next_obs = torch.tensor(batch, dtype=torch.float32, device=self.device)
                #next_obs = torch.Tensor(next_obs_np).to(self.device)
                next_done = torch.Tensor(next_done).to(self.device)

                if pbar is not None:
                    pbar.update(1)

                # Logging returns
                if "episode" in infos and "_episode" in infos:
                    for i, done in enumerate(infos["_episode"]):
                        if done:
                            ep_return = float(infos["episode"]["r"][i])
                            ep_length = int(infos["episode"]["l"][i])
                            #print(f"global_step={self.global_step}, episodic_return={ep_return}, episode_length={ep_length}")
                            self.writer.add_scalar("charts/episodic_return", ep_return, self.global_step)
                            self.writer.add_scalar("charts/episodic_length", ep_length, self.global_step)

                            # CSV logging
                            self.episode_counter += 1
                            self.csv_writer.writerow([self.episode_counter, ep_return, self.global_step])
                            self.csv_file.flush()  # ensure the file is updated incrementally

            # Compute GAE & returns
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # Flatten the batch
            #b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            # Flatten the batch to (batch_size, obs_dim)
            b_obs = self.obs.reshape(self.batch_size, -1)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimize policy + value
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # approximate KL for early stopping
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy
                    entropy_loss = entropy.mean()

                    # Final loss
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            # Logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
            sps = int(self.global_step / (time.time() - start_time))
            self.writer.add_scalar("charts/SPS", sps, self.global_step)
            #print(f"Iteration={iteration}/{self.num_iterations}, global_step={self.global_step}, SPS={sps}")

        if pbar is not None:
            pbar.close()
        self.envs.close()
        self.writer.close()
        self.csv_file.close()
        print("Training finished!")

    # ------------------------------------------------------------------
    # Hyperparameter export
    # ------------------------------------------------------------------
    def _export_hparams(self):
        return dict(
            #env_id        = self.env_id,
            seed          = self.seed,
            agent_type    = self.agent_type,
            learning_rate = self.learning_rate,
            num_envs      = self.num_envs,
            num_steps     = self.num_steps,
            gamma         = self.gamma,
            gae_lambda    = self.gae_lambda,
            norm_adv      = self.norm_adv,
            clip_coef     = self.clip_coef,
            clip_vloss    = self.clip_vloss,
            ent_coef      = self.ent_coef,
            vf_coef       = self.vf_coef,
            max_grad_norm = self.max_grad_norm,
            target_kl     = self.target_kl,
        )
    
    # ──────────────────────────────────────────────────────────────
    # Persistence helpers: secure (JSON+zip) + fallback (pickle)
    # ──────────────────────────────────────────────────────────────
    def save(self, file_path: str, *, metadata: dict | None = None):
        tmp_dir = tempfile.mkdtemp()
        # 1) tensors
        torch.save(self.agent.state_dict(),     f"{tmp_dir}/policy.pth")
        torch.save(self.optimizer.state_dict(), f"{tmp_dir}/optim.pth")

        # 2) agent_config safe dump
        try:
            json.dumps(self.agent_config)
            with open(f"{tmp_dir}/agent_cfg.json", "w") as fcfg:
                json.dump(self.agent_config, fcfg)
            cfg_fmt = "json"
        except TypeError:
            with open(f"{tmp_dir}/agent_cfg.pkl", "wb") as fcfg:
                cloudpickle.dump(self.agent_config, fcfg)
            cfg_fmt = "pkl"

        # 3) meta
        meta = {
            "global_step": self.global_step,
            "hparams":     self._export_hparams(),
            "cfg_format":  cfg_fmt,
            **(metadata or {})
        }
        with open(f"{tmp_dir}/meta.json", "w") as f:
            json.dump(meta, f)

        shutil.make_archive(file_path, "zip", tmp_dir)
        shutil.rmtree(tmp_dir)
        print(f"Saved checkpoint: {file_path}.zip")

    @classmethod
    def load(cls, file_path: str, env=None, device: str = "cpu", allow_pickle: bool = False):
        if not file_path.endswith(".zip"):
            file_path += ".zip"

        with zipfile.ZipFile(file_path, "r") as zf:
            # 1) Read primitive metadata
            meta    = json.loads(zf.read("meta.json"))
            hparams = meta["hparams"].copy()

            # 2) Load agent_config *before* building PPO
            fmt = meta.get("cfg_format", "json")
            if fmt == "json":
                cfg = json.loads(zf.read("agent_cfg.json"))
            else:
                if not allow_pickle:
                    raise RuntimeError(
                        "Checkpoint uses pickle for agent_config; pass allow_pickle=True to load it."
                    )
                cfg = cloudpickle.loads(zf.read("agent_cfg.pkl"))

            # 3) Assemble constructor kwargs, now including agent_config
            kwargs = dict(
                env           = env,
                run_name      = "PPO-loaded",
                agent_config  = cfg,
                **hparams,    # includes agent_type, seed, net hyperparams, etc.
            )

            # 4) Build the PPO (this calls build_agent with the restored cfg)
            obj = cls(**kwargs)

            # 5) Restore weights & optimizer
            obj.agent.load_state_dict(
                torch.load(BytesIO(zf.read("policy.pth")), map_location=device)
            )
            obj.optimizer.load_state_dict(
                torch.load(BytesIO(zf.read("optim.pth")), map_location=device)
            )
            obj.global_step = meta["global_step"]

        print(f"Loaded checkpoint: {file_path}")
        return obj

    # unsafe full-pickle helpers
    def save_full(self, path: str):
        torch.save(self.__dict__, path, _use_new_zipfile_serialization=False)
        print(f"Full-pickle checkpoint written to {path} (trusted use only)")

    @classmethod
    def load_full(cls, path: str, **kw):
        return torch.load(path, weights_only=False, **kw)

    def close(self):
        self.envs.close()
        self.writer.close()
        self.csv_file.close()