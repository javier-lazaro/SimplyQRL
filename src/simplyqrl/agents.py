import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
import pennylane as qml
import gymnasium as gym


from .qlayers import build_basic_qlayer, build_hsiao_qlayer, build_skolik_qlayer, build_dr_qlayer
from .embeddings import build_embedding_fn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal layer initialization as used in CleanRL."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

"""
Expected configs:
- mlp: 
    "net_arch": {"pi": [64,64], "vf": [64,64]},
    "activation": nn.ReLU / nn.Tanh

- hybrid:
    "net_arch": {"pi": [4], "vf": [4]}, 
    "activation": nn.ReLU,
    "n_qubits": 4,
    "n_layers_q": 2
"""

def build_agent(type, obs_shape, n_actions, config, is_qnet=False):
        """
        Build the default CleanRL MLP agent with separate actor & critic networks
        in a single module (as in the original code).
        """
        type = type.lower()
        if type == "mlp":
            if config is None: 
                config = {
                    "net_arch": [120,84] if is_qnet else {"pi": [64], "vf": [64]},
                    "activation": nn.ReLU if is_qnet else nn.Tanh,
                }
            obs_dim = int(np.prod(obs_shape))
            return ClassicalAgent(obs_dim, n_actions, config, is_qnet)
        
        elif type == "hybrid":
            if config is None:
                config = {
                    "net_arch": [4] if is_qnet else {"pi": [4], "vf": [4]}, 
                    "activation": nn.Tanh,
                    "n_qubits": 4,
                    "n_layers_q": 2
                }
            obs_dim = int(np.prod(obs_shape))
            return HybridAgent(obs_dim, n_actions, config, is_qnet)


class OutputReuse(nn.Module):
    def __init__(self, n_repeats: int):
        """
        Args:
            n_repeats (int): How many times to repeat the quantum layer's output.
        """
        super(OutputReuse, self).__init__()
        self.n_repeats = n_repeats

    def forward(self, x):
        # x is assumed to be of shape [batch_size, n_qubits]
        # This repeats the features along the feature dimension.
        return x.repeat(1, self.n_repeats)


class SelectiveOutputReuse(nn.Module):
    def __init__(self, n_repeats: int, indices: list):
        """
        Args:
            n_repeats (int): Number of times to repeat the selected features.
            indices (list): List of indices to select from the input tensor.
        """
        super(SelectiveOutputReuse, self).__init__()
        self.n_repeats = n_repeats
        self.indices = indices

    def forward(self, x):
        # For safety, ensure x is 2D: [batch_size, features].
        if x.dim() == 1:
            x = x.unsqueeze(0)
        selected = x[:, self.indices]  # should reduce from 4 to 3 features.
        repeated = selected.repeat(1, self.n_repeats)  # should convert (batch, 3) to (batch, 3*n_repeats)
        # Debug: Uncomment the next line to check shapes.
        #print("Input:", x.shape, "Selected:", selected.shape, "Repeated:", repeated.shape)
        return repeated


class OutputScale(nn.Module):
    """
    Multiplies each action-value by a trainable scalar. 
    If x has shape (B, act_dim) the result has the same shape.
    """
    def __init__(self, act_dim, init_value: float = 2.0):
        """
        Args:
            act_dim: Dimension of the action space.
            init_value (float): Initial value for the parameter.
        """
        super(OutputScale, self).__init__()
        self.scale = nn.Parameter(
            torch.full((act_dim,), init_value, dtype=torch.float32)
        )

    def forward(self, x):
        return x * self.scale


class ClassicalAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, config, is_qnet=False):
        super().__init__()

        net_arch = config.get("net_arch", {"pi": [64,64], "vf": [64,64]})
        if isinstance(net_arch, dict):
            # Already provided separately for "pi" and "vf"
            pi_arch = net_arch.get("pi")
            vf_arch = net_arch.get("vf")
        elif isinstance(net_arch, list):
            # A single list: assume it's the same for both "pi" and "vf"
            pi_arch = vf_arch = net_arch
        else:
            raise ValueError("net_arch must be either a dict or a list")

        activation = config.get("activation", nn.ReLU)

        # Check for selective output reuse configuration.
        # If 'reuse_indices' is provided, use the SelectiveOutputReuse module.
        self.n_repeats = config.get("n_repeats", 1)
        self.reuse_indices = config.get("reuse_indices", None)
        if self.n_repeats > 1 and self.reuse_indices is not None:
            self.input_reuse = SelectiveOutputReuse(self.n_repeats, self.reuse_indices)
            input_dim = len(self.reuse_indices) * self.n_repeats
        elif self.n_repeats > 1:
            self.input_reuse = OutputReuse(self.n_repeats)
            input_dim = obs_dim * self.n_repeats
        else:
            self.input_reuse = None
            input_dim = obs_dim

        self.is_qnet = is_qnet

        if self.is_qnet:
            # Build Q Network.
            net_layers = []
            prev_dim = input_dim
            for idx, hidden_size in enumerate(pi_arch):
                layer = nn.Linear(prev_dim, hidden_size)
                net_layers.append(layer)
                net_layers.append(activation())
                prev_dim = hidden_size
            # Final Actor Layer
            last_layer = nn.Linear(prev_dim, act_dim)
            net_layers.append(last_layer)
            self.network = nn.Sequential(*net_layers)
            #print(f"BUILT Q-Network: {self.network}")

        else:
            # Build Critic network.
            critic_layers = []
            prev_dim = input_dim
            for idx, hidden_size in enumerate(vf_arch):
                layer = layer_init(nn.Linear(prev_dim, hidden_size))
                critic_layers.append(layer)
                critic_layers.append(activation())
                prev_dim = hidden_size
            # Final Critic layer
            last_layer = layer_init(nn.Linear(prev_dim, 1), std=1.0)
            critic_layers.append(last_layer)
            self.critic = nn.Sequential(*critic_layers)
            #print(f"BUILT CRITIC: {self.critic}")

            # Build Actor network.
            actor_layers = []
            prev_dim = input_dim
            for idx, hidden_size in enumerate(pi_arch):
                layer = layer_init(nn.Linear(prev_dim, hidden_size))
                actor_layers.append(layer)
                actor_layers.append(activation())
                prev_dim = hidden_size
            # Final Actor Layer
            last_layer = layer_init(nn.Linear(prev_dim, act_dim), std=0.01)
            actor_layers.append(last_layer)
            self.actor = nn.Sequential(*actor_layers)
            #print(f"BUILT ACTOR: {self.actor}")

    def get_agent_info(self):
        print("\nAGENT INFORMATION:")

        if self.is_qnet:
            print(f"* Agent type: Q-Network (Classical)")
            print(f"* Number of trainable parameters: {sum(p.numel() for p in self.network.parameters())}")
        else:
            print(f"* Agent type: Actor-Critic (Classical)")
            print(f"* Number of trainable parameters: {sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())}")

        print("* Network configuration:\n")
        print(self)

    def forward(self, x):
        if self.input_reuse is not None:
            x = self.input_reuse(x)
        return self.network(x)

    def get_value(self, x):
        if self.input_reuse is not None:
            x = self.input_reuse(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # Always apply the input reuse transformation.
        if self.input_reuse is not None:
            x = self.input_reuse(x)
        # Debug: check the shape after transformation.
        # Uncomment the next line if you need to debug.
        #print("Shape after reuse:", x.shape)
        
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # Use the same transformed x for the critic as well.
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

class HybridAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, config, is_qnet=False):
        """
        A hybrid actor-critic agent where each network
        (actor and critic) begins with a quantum layer
        and then has classical layers.
        
        Args:
            obs_dim (int): Dimension of observations.
            act_dim (int): Dimension of action space.
            config (dict): Configuration dictionary.
                Expects keys like:
                  - "circ_type": specific quantum circuit type: "hsiao", "skolik" or "custom"
                  - "n_qubits": how many qubits for the quantum circuit
                  - "n_layers": how many layers in the quantum circuit
                  - "reuse_repetitions": how many times we wish to reuse the output of the quantum circuit
                  - "embedding": type of embedding to apply to the circuit (will be ignored for non custom circuits)
                  - "embedding_kwargs": additional arguments for the embedding (will be ignored for non custom circuits)
                  - "emb_indices": indices in the obs Tensor to apply the embeddding. 
                  - "transformation_fn": transformation function to apply to the observation data prior to embedding. 
                  - "net_arch": can be dict {"pi": [...], "vf": [...]} or a single list
                  - "activation": an nn.Module class, default nn.ReLU
        """
        super().__init__()

        # Parse network architecture
        self.net_arch = config.get("net_arch", {"pi": [], "vf": []})
        if isinstance(self.net_arch, dict):
            self.pi_arch = self.net_arch.get("pi", [])
            self.vf_arch = self.net_arch.get("vf", [])
        elif isinstance(self.net_arch, list):
            self.pi_arch = self.vf_arch = self.net_arch
        else:
            raise ValueError("net_arch must be either a dict or a list")

        self.is_qnet = is_qnet
        self.activation = config.get("activation", nn.ReLU)
        self.n_qubits = config.get("n_qubits", 4)
        self.n_layers_q = config.get("n_layers_q", 1)
        self.ent = config.get("ent", False)

        # 1. Build the embedding function.
        # The user can either provide a string to select a built-in embedding
        # (e.g. "angle", "amplitude", "triple_rotation") along with extra kwargs,
        # or supply a custom callable.
        embedding_option = config.get("embedding", "angle")
        embedding_kwargs = config.get("embedding_kwargs", {})
        embedding_fn = build_embedding_fn(embedding_option, embedding_kwargs)

        self.circ_type = config.get("circ_type", "custom")
        self.reuse_repetitions = config.get("reuse_repetitions", 1)  # e.g. 3 for tripling the output
        self.indices = config.get("emb_indices", None)
        self.transform_fn = config.get("transform_fn", None)

        # Check if the agent has a Q-Network or Actor-Critic behaviour
        if is_qnet:
            # 2. Build quantum layer for the Q-Network.
            if self.circ_type == "hsiao":
                hsiao_emb_type = config.get("hisao_emb_type", "multi")
                self.qnet_q_layer = build_hsiao_qlayer(self.n_qubits, 
                                                       self.n_layers_q, 
                                                       emb_type=hsiao_emb_type, 
                                                       emb_indices=self.indices, 
                                                       transform_fn=self.transform_fn,
                                                       ent=self.ent)
                
            elif self.circ_type == "skolik":
                self.qnet_q_layer = build_skolik_qlayer(self.n_qubits, 
                                                        self.n_layers_q, 
                                                        emb_indices=self.indices, 
                                                        transform_fn=self.transform_fn,
                                                        ent=self.ent)
                
            elif self.circ_type == "dr":
                self.qnet_q_layer = build_dr_qlayer(self.n_qubits, 
                                                    self.n_layers_q, 
                                                    emb_indices=self.indices, 
                                                    transform_fn=self.transform_fn,
                                                    ent=self.ent)
            else:
                self.qnet_q_layer = build_basic_qlayer(embedding_fn, self.n_qubits, self.n_layers_q)
            
            # 3. Build the classical part of the Q-Network.
            qnet_layers = []
            qnet_layers.append(self.qnet_q_layer)

            # Insert the output reuse layer if desired.
            if self.reuse_repetitions > 1:
                qnet_layers.append(OutputReuse(self.reuse_repetitions))
                # The effective input dimension to the classical network increases.
                prev_dim = self.n_qubits * self.reuse_repetitions
            else:
                prev_dim = self.n_qubits # Output dimension from the quantum layer

            for size in self.pi_arch:
                qnet_layers.append(nn.Linear(prev_dim, size))
                qnet_layers.append(self.activation())
                prev_dim = size
            # Final Q-Net layer: output logits for each action.
            qnet_layers.append(nn.Linear(prev_dim, act_dim))
            self.network = nn.Sequential(*qnet_layers)

            # Insert the output scaling layer if desired.
            if config.get("use_output_scaling", False):
                init_val = config.get("output_scale_init", 2.0)
                qnet_layers.append(OutputScale(act_dim, init_val))

        # If the behaviour is Actor-Critic we create two separate networks.
        else:
            # 2. Build quantum layer for the critic.
            if self.circ_type == "hsiao":
                hsiao_emb_type = config.get("hisao_emb_type", "multi")
                self.critic_q_layer = build_hsiao_qlayer(self.n_qubits, 
                                                         self.n_layers_q, 
                                                         emb_type=hsiao_emb_type, 
                                                         emb_indices=self.indices,
                                                         transform_fn=self.transform_fn,
                                                         ent=self.ent)
            elif self.circ_type == "skolik":
                self.critic_q_layer = build_skolik_qlayer(self.n_qubits, 
                                                          self.n_layers_q, 
                                                          emb_indices=self.indices,
                                                          transform_fn=self.transform_fn, 
                                                          ent=self.ent)
            elif self.circ_type == "dr":
                self.critic_q_layer = build_dr_qlayer(self.n_qubits, 
                                                      self.n_layers_q, 
                                                      transform_fn=self.transform_fn,
                                                      ent=self.ent)
            else:
                self.critic_q_layer = build_basic_qlayer(embedding_fn, 
                                                         self.n_qubits, 
                                                         self.n_layers_q)
            
            # 3. Build the classical part of the critic.
            critic_layers = []
            # Quantum layer as the first block.
            critic_layers.append(self.critic_q_layer)

            # Insert the output reuse layer if desired.
            if self.reuse_repetitions > 1:
                critic_layers.append(OutputReuse(self.reuse_repetitions))
                # The effective input dimension to the classical network increases.
                prev_dim = self.n_qubits * self.reuse_repetitions
            else:
                prev_dim = self.n_qubits # Output dimension from the quantum layer

            for size in self.vf_arch:
                critic_layers.append(layer_init(nn.Linear(prev_dim, size)))
                critic_layers.append(self.activation())
                prev_dim = size
            # Final output layer for the critic.
            critic_layers.append(layer_init(nn.Linear(prev_dim, 1), std=1.0))
            self.critic = nn.Sequential(*critic_layers)
            
            # 4. Build quantum layer for the actor.
            if self.circ_type == "hsiao":
                self.actor_q_layer = build_hsiao_qlayer(self.n_qubits, 
                                                        self.n_layers_q, 
                                                        emb_type=hsiao_emb_type, 
                                                        emb_indices=self.indices,
                                                        transform_fn=self.transform_fn, 
                                                        ent=self.ent)
            elif self.circ_type == "skolik":
                self.actor_q_layer = build_skolik_qlayer(self.n_qubits, 
                                                         self.n_layers_q,
                                                         emb_indices=self.indices, 
                                                         transform_fn=self.transform_fn,
                                                         ent=self.ent)
            elif self.circ_type == "dr":
                self.actor_q_layer = build_dr_qlayer(self.n_qubits, 
                                                     self.n_layers_q,
                                                     emb_indices=self.indices, 
                                                     transform_fn=self.transform_fn,
                                                     ent=self.ent)
            else:
                self.actor_q_layer = build_basic_qlayer(embedding_fn, 
                                                        self.n_qubits, 
                                                        self.n_layers_q)
            
            # 5. Build the classical part of the actor.
            actor_layers = []
            actor_layers.append(self.actor_q_layer)

            # Insert the output reuse layer if desired.
            if self.reuse_repetitions > 1:
                actor_layers.append(OutputReuse(self.reuse_repetitions))
                # The effective input dimension to the classical network increases.
                prev_dim = self.n_qubits * self.reuse_repetitions
            else:
                prev_dim = self.n_qubits # Output dimension from the quantum layer

            for size in self.pi_arch:
                actor_layers.append(layer_init(nn.Linear(prev_dim, size)))
                actor_layers.append(self.activation())
                prev_dim = size
            # Final actor layer: output logits for each action.
            actor_layers.append(layer_init(nn.Linear(prev_dim, act_dim), std=1.0))
            self.actor = nn.Sequential(*actor_layers)


    def get_agent_info(self, env=None):
        # Access the underlying QNode from the TorchLayer (qlayer_actor in this case)
        # qlayer_actor.qnode is the QNode object
        from pennylane import draw

        if env is not None:
            sample = env.observation_space.sample()
            sample_inp = torch.as_tensor(sample).unsqueeze(0)
        else:
            obs_dim = getattr(self, "obs_dim", 1)
            sample_inp = torch.zeros((1, obs_dim))

        if self.circ_type == "skolik":
            sample_weights = torch.randn(self.n_layers_q, 2, self.n_qubits)  
        elif self.circ_type == "dr":
            sample_weights = torch.randn(self.n_layers_q, 3, self.n_qubits)  
        else:
            sample_weights = torch.randn(self.n_layers_q, self.n_qubits)

        print("\nAGENT INFORMATION:")

        if self.is_qnet:
            print(f"* Agent type: Q-Network (Hybrid)")
            print(f"* Number of quantum parameters: {sum(p.numel() for p in self.qnet_q_layer.parameters())}")
            print(f"* Number of trainable parameters: {sum(p.numel() for p in self.network.parameters())}")
            print(f"* Quantum circuit:\n")
            print(f"SAMPLE INPUT: {sample_inp}\n")

            print(draw(self.qnet_q_layer.qnode, level="device")(
                inputs=sample_inp, 
                weights=sample_weights
                ))
             
            # Print the names and shapes of all parameters
            for name, param in self.qnet_q_layer.named_parameters():
                print(name, param.shape)

        else:
            print(f"* Agent type: Actor-Critic (Hybrid)")
            print(f"* Number of quantum parameters: {sum(p.numel() for p in self.actor_q_layer.parameters()) + sum(p.numel() for p in self.critic_q_layer.parameters())}")
            print(f"* Number of trainable parameters: {sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())}")
            print(f"* Quantum circuit:\n")
            print(f"SAMPLE INPUT: {sample_inp}\n")

            print("Actor Q-Node:")
            print(draw(self.actor_q_layer.qnode, level="device")(
                inputs=sample_inp, 
                weights=sample_weights
                ))
            for name, param in self.actor_q_layer.named_parameters():
                print(name, param.shape, end="\n\n")
            
            print("Critic Q-Node:")
            print(draw(self.critic_q_layer.qnode, level="device")(
                inputs=sample_inp, 
                weights=sample_weights
                ))            
            for name, param in self.critic_q_layer.named_parameters():
                print(name, param.shape, end="\n\n")

        print("\n* Network configuration:\n")
        # Print the whole model
        print(self)


    def forward(self, x):
        # Used for Q-Network. To maintain consistent behaviour with base CleanRL.
        """
        # Unbatched operation
        if x.dim() == 1:                 
            x = x.unsqueeze(0)           

        outs = []
        for xi in x:                      
            qi = self.network(xi.unsqueeze(0))   # (1, act_dim)
            outs.append(qi)

        return torch.cat(outs, dim=0)     # (B, act_dim)
        """

        # Batched operation
        return self.network(x)

    def get_value(self, x):
        # Evaluate the critic
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Function to obtain the action probability distribution and critic values. 
        """

        """
        # Unbatched operation
        def _single_forward(xi, ai=None):
            logits_i = self.actor(xi)           # (1, act_dim)
            d_i      = Categorical(logits=logits_i)
            ai = d_i.sample() if ai is None else ai
            return (
                ai,                             # (1,)
                d_i.log_prob(ai),               # (1,)
                d_i.entropy(),                  # (1,)
                self.critic(xi)                 # (1, 1)
            )
    
        if x.dim() == 1:                # make sure we can iterate
            x = x.unsqueeze(0)          # (1, obs_dim)

        lst = []
        for i, xi in enumerate(x):      # type: ignore
            xi = xi.unsqueeze(0)        # (1, obs_dim)
            ai = None
            if action is not None:      # user supplied a tensor of actions
                ai = action[i].unsqueeze(0)
            lst.append(_single_forward(xi, ai))

        # collate lists → tensors; keeps autograd graph intact
        a, lp, ent, v = zip(*lst)           # tuples of (1,) or (1,1) tensors
        actions    = torch.cat(a,   dim=0)  # (B,)
        log_probs  = torch.cat(lp,  dim=0)  # (B,)
        entropies  = torch.cat(ent, dim=0)  # (B,)
        values     = torch.cat(v,   dim=0)  # (B, 1)
        
        return actions, log_probs, entropies, values
        """

        # Batched operation
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)