import numpy as np
import torch
import math

def linear_normalize(x, in_min, in_max, out_min, out_max):
    """
    Normalizes x from the interval [in_min, in_max] to [out_min, out_max].
    """
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

class ObservationTransformer:
    def __call__(self, obs):
        raise NotImplementedError


class IdentityTransformer:
    """No-op transform."""
    def __call__(self, data):
        return data


class ArctanTransformer:
    """Example transform that uses arctan (just as an example)."""
    def __call__(self, data):
        return torch.atan(data)


class CartPoleNormalizationTransformer:
    """
    Custom transformer for observations with four elements.
    
    For a single observation [A, B, C, D] or a batch of observations,
    applies the following:
      - First element: Normalize from [-4.8, 4.8] to [-π, π].
      - Second element: Compute 2 * atan(x).
      - Third element: Normalize from [-0.418, 0.418] to [-π, π].
      - Fourth element: Compute 2 * atan(x).
    """
    def __call__(self, data):
        # Ensure data is a torch.Tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)
            
        if data.dim() == 1:
            # Single observation of shape [4]
            transformed = data.clone()
            # Normalize first element from [-4.8, 4.8] to [-pi, pi]
            transformed[0] = linear_normalize(data[0], -4.8, 4.8, -math.pi, math.pi)
            # Second element: 2 * atan(x)
            transformed[1] = 2 * torch.atan(data[1])
            # Normalize third element from [-0.418, 0.418] to [-pi, pi]
            transformed[2] = linear_normalize(data[2], -0.418, 0.418, -math.pi, math.pi)
            # Fourth element: 2 * atan(x)
            transformed[3] = 2 * torch.atan(data[3])
            return transformed

        elif data.dim() == 2:
            # Batched observations: shape [batch_size, 4]
            transformed = data.clone()
            # Normalize first column
            transformed[:, 0] = linear_normalize(data[:, 0], -4.8, 4.8, -math.pi, math.pi)
            # Apply 2*atan to second column
            transformed[:, 1] = 2 * torch.atan(data[:, 1])
            # Normalize third column
            transformed[:, 2] = linear_normalize(data[:, 2], -0.418, 0.418, -math.pi, math.pi)
            # Apply 2*atan to fourth column
            transformed[:, 3] = 2 * torch.atan(data[:, 3])
            return transformed

        else:
            raise ValueError("Data must be a 1D or 2D tensor with four elements per observation.")


class FrozenNormalizationTransformer(ObservationTransformer):
    """
    Normalize the discrete FrozenLake state s ∈ [0, rows*cols - 1]
    into an angle in [0, 2π], so it's ready for AngleEmbedding.
    """
    def __init__(self, grid_size: str):
        # parse "NxM"
        rows, cols = map(int, grid_size.lower().split('x'))
        self.n_states = rows * cols
        self.out_max = 2*math.pi

    def __call__(self, data):
        # ensure torch Tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)

        # single sample: shape [1]
        if data.dim() == 1:
            transformed = data.clone()
            # data[0] is the discrete state
            transformed[0] = linear_normalize(data[0], 0, self.n_states - 1, 0, self.out_max)
            return transformed

        # batched: shape [batch_size, 1]
        elif data.dim() == 2 and data.shape[1] == 1:
            transformed = data.clone()
            transformed[:, 0] = linear_normalize(data[:, 0], 0, self.n_states - 1, 0, self.out_max)
            return transformed

        else:
            raise ValueError(
                f"FrozenNormalizationTransformer expects data shape [1] or [B,1], got {tuple(data.shape)}"
            )
        

class FrozenBasisToAngleTransformer(ObservationTransformer):
    """
    Convert each discrete FrozenLake state s ∈ [0, rows*cols - 1]
    into its bitstring on n_qubits = ceil(log2(rows*cols)),
    then map bits {0,1} → angles {0, π}.
    """
    def __init__(self, grid_size: str):
        rows, cols = map(int, grid_size.lower().split('x'))
        n_states = rows * cols
        self.n_qubits = math.ceil(math.log2(n_states))

    def __call__(self, data):
        # ensure torch LongTensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.long)
        else:
            data = data.long()

        # single sample: shape [1]
        if data.dim() == 1:
            s = int(data[0])
            bits = torch.tensor(
                [(s >> i) & 1 for i in range(self.n_qubits)],
                dtype=torch.float,
                device=data.device
            )
            return bits * math.pi  # shape [n_qubits]

        # batched: shape [batch_size, 1]
        elif data.dim() == 2 and data.shape[1] == 1:
            s = data[:, 0]  # shape [batch_size]
            # build tensor of shape [batch_size, n_qubits]
            bit_indices = torch.arange(self.n_qubits, device=data.device).unsqueeze(0)
            bits = ((s.unsqueeze(1) >> bit_indices) & 1).float() 
            return bits * math.pi  # shape [batch_size, n_qubits]

        else:
            raise ValueError(
                f"FrozenBasisToAngleTransformer expects data shape [1] or [B,1], got {tuple(data.shape)}"
            )