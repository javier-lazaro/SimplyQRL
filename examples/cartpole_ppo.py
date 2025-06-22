from simplyqrl.ppo import PPO
from simplyqrl.transformations import CartPoleNormalizationTransformer, ArctanTransformer
import gymnasium as gym
import numpy as np
import torch
import os 

# Dependancies for recoding video only
#import imageio
#from PIL import Image       # Technically not needed if resize_factor is None or 1


def evaluate_agent(
    trained_agent,
    num_episodes: int = 10,
    record_video: bool = False,
    video_path: str = "videos/eval_agent.mp4",
    fps: int = 10,
    resize_factor: int | None = None,
):
    """Evaluate a single PPO agent on FrozenLake and record the rollouts in a single compiled video.

    Args:
        num_episodes: How many episodes to run.
        video_path: Destination MP4 (episodes concatenated).
        fps: Frames per second of the output video.
        resize_factor: If given, each captured 256x256 frame is up-scaled by this
            integer factor (using *nearest-neighbour* to keep the crisp grid look).
            E.g. 'resize_factor=4' => 1024x1024 video.

    Returns:
        mean_return (float): Average episodic reward over *num_episodes*.
    """

    returns: list[float] = [] # Initialise returns list

    # 1. Fresh enviroment - with render env (rgb_array) for recording if needed
    if record_video:
        env = gym.make("CartPole-v1", render_mode="rgb_array")

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        frames: list[np.ndarray] = []
    else:
        env = gym.make("CartPole-v1")

    
    print(f"\nCommencing evaluation of the trained agent in the CartPole-v1 enviroment:\n")

    # 2. Evaluation loop
    agent = trained_agent.agent
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0) 
            with torch.no_grad():
                action_tensor, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = int(action_tensor.item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward

            if record_video:
                frame = env.render()
                if resize_factor and resize_factor > 1:
                    img = Image.fromarray(frame)
                    w, h = img.size
                    img = img.resize((w * resize_factor, h * resize_factor), resample=Image.NEAREST)
                    frame = np.array(img)
                frames.append(frame)
        returns.append(ep_ret)

    # 3. Save video (if needed) & report stats
    mean_return = sum(returns) / len(returns)
    
    if record_video:
        writer = imageio.get_writer(video_path, fps=fps, format="ffmpeg")
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"Video saved to {video_path} (resolution: {frames[0].shape[1]}x{frames[0].shape[0]})")

    print(f"Mean return over {num_episodes} episodes: {mean_return:.2f}")
   
    env.close()
    return mean_return


if __name__ == "__main__": 

    # In this example we present the training and evaluation of a Hybrid agent to solve the CartPole-v1 enviroment 
    # The resolution process followed in this example can be replicated for other versions of CartPole
    

    # ENVIROMENT CREATION
    env = gym.make("CartPole-v1")

    # HYBRID AGENT CONFIGURATION - SKOLIK + DR
    config_hybrid = {
        "circ_type": "skolik",                                  # Uses the Skolik et al. PQC architecture fror the quantum layer
        "n_qubits": 4,                                          # Circuit uses 4 Qubits
        "n_layers_q": 5,                                        # 5 Layers of DR are applied
        "ent": True,                                            # Entanglement is left ON (Default for Skolik et al.) 
        "transform_fn": CartPoleNormalizationTransformer(),     # Normalization function for CartPole, found under transformations.py
        #"net_arch": [4],                                       # A final classical NN architecture could be defined here
    }

    ## EXPERIMENT 1: SOLVING THE ENV USING SKOLIK ET AL. + DATA REUPLOADING -- COMMENT THIS BLOCK IF YOU JUST WANT TO EVALUATE THE RESULT ## 


    # Algorithm setup
    seed = 1                    # Seed for reproducibility                   
    run_name = "PPO_Cart_sk"    # Name used for logs
    ppo = PPO(env=env, agent_type="hybrid", agent_config=config_hybrid, run_name=run_name, seed=seed) # Hyperparameters can be set via directly in the constructor
    ppo.agent.get_agent_info(env)  # Shows the full information of the agent

    # Training and saving the agent
    ppo.train(total_timesteps=1_000, progress_bar=True)   # Main training loop
    ppo.save("checkpoints/PPO/CP_hybrid_sk")                # Saving the trained agent
    ppo.close()  


    ## EVAUATING THE AGENT -- COMMENT THIS BLOCK IF YOU JUST WANT TO TRAIN IT ##

    # Loading the agent after training
    checkpoint_path = "checkpoints/PPO/CP_hybrid_sk"                    # If you saved it another location, adjust the path
    ppo_loaded = PPO.load(checkpoint_path, env=env, allow_pickle=True)  # Allow pickle set to true because CartPoleNormalizationTransformer was used in config
    ppo_loaded.agent.get_agent_info(env)

    # Evaluating the agent (No video, just stats)
    evaluate_agent(ppo_loaded, num_episodes=10, record_video=False)

    # Evaluating the agent (Video + stats) --> See additional dependancies at the top for this to work
    #evaluate_agent(ppo_loaded, num_episodes=10, record_video=True, video_path="videos/ppo_cart_sk.mp4", fps=10, resize_factor=4)

    ppo_loaded.close()

    

    # HYBRID AGENT CONFIGURATION - HSIAO + OR
    config_hybrid = {
        "circ_type": "hsiao",                                   # Uses the Hsiao et al. PQC architecture for the quantum layer
        "n_qubits": 4,                                          # Circuit uses 4 Qubits
        "emb_indices": [1,2,3],                                 # Observation array indices to use during embedding - Discard cartPos (as in the original paper)
        "reuse_repetitions": 16,                                # Output is reused 16 times
        "ent": False,                                           # Entanglement is left OFF (Default for Hsiao et al.) 
        "transform_fn": ArctanTransformer(),                    # Generic arctan normalizaton, found under transformations.py
    }

    ## EXPERIMENT 2: SOLVING THE ENV USING HSIAO ET AL. + OUTPUT REUSE -- COMMENT THIS BLOCK IF YOU JUST WANT TO EVALUATE THE RESULT ## 

    # Algorithm setup
    seed = 1                    # Seed for reproducibility                   
    run_name = "PPO_Cart_hs"    # Name used for logs
    ppo = PPO(env=env, agent_type="hybrid", agent_config=config_hybrid, run_name=run_name, seed=seed) # Hyperparameters can be set via directly in the constructor
    ppo.agent.get_agent_info(env)  # Shows the full information of the agent
 
    # Training and saving the agent
    ppo.train(total_timesteps=100_000, progress_bar=True)   # Main training loop
    ppo.save("checkpoints/PPO/CP_hybrid_hs")                # Saving the trained agent
    ppo.close() 

    # Loading the agent after training
    checkpoint_path = "checkpoints/PPO/CP_hybrid_hs"                    # If you saved it another location, adjust the path
    ppo_loaded = PPO.load(checkpoint_path, env=env, allow_pickle=True)  # Allow pickle set to true because ArctanTransformer was used in config

    # Evaluating the agent (No video, just stats)
    evaluate_agent(ppo_loaded, num_episodes=10, record_video=False)

    # Evaluating the agent (Video + stats) --> See additional dependancies at the top for this to work
    #evaluate_agent(ppo_loaded, num_episodes=20, record_video=True, video_path="videos/ppo_cat_hs.mp4", fps=10, resize_factor=4)

    ppo_loaded.close()