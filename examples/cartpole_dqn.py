from simplyqrl.dqn import DQN
from simplyqrl.transformations import CartPoleNormalizationTransformer
import gymnasium as gym
import numpy as np
import torch
import os 


if __name__ == "__main__": 

    # In this example we present the training and evaluation of a Hybrid agent to solve the CartPole-v1 enviroment 
    # The resolution process followed in this example can be replicated for other versions of CartPole

    # ENVIROMENT CREATION
    env = gym.make("CartPole-v1")

    # HYBRID AGENT CONFIGURATION - SKOLIK + DR
    config_hybrid = {
        "circ_type": "skolik",                                  # Uses the Skolik et al. PQC architecture fror the quantum layer
        "n_qubits": 8,                                          # Circuit uses 8 Qubits
        "n_layers_q": 5,                                        # 5 Layers of DR are applied
        "ent": True,                                            # Entanglement is left ON (Default for Skolik et al.) 
        "transform_fn": CartPoleNormalizationTransformer(),     # Normalization function for CartPole, found under transformations.py
        "net_arch": [4],                                        # A final classical NN architecture could be defined here
        "activation": torch.nn.Identity                         # An Identity (Linear) activation function is used
    }

    ## TRAINING THE AGENT -- COMMENT THIS BLOCK IF YOU JUST WANT TO EVALUATE THE RESULT ## 
    
    # Algorithm setup
    seed = 1                    # Seed for reproducibility                   
    run_name = "DQN_Cart_sk"    # Name used for logs
    dqn = DQN(env=env, agent_type="hybrid", agent_config=config_hybrid, run_name=run_name, seed=seed) # Hyperparameters can be set via directly in the constructor
    dqn.agent.get_agent_info(env)  # Shows the full information of the agent


    # Training and saving the agent
    #dqn.train(total_timesteps=100_000, progress_bar=True)   # Main training loop
    #dqn.save("checkpoints/DQN/CP_hybrid_sk")                # Saving the trained agent
    #dqn.close()  