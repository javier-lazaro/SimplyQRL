import pennylane as qml
import numpy as np
from functools import partial

def build_embedding_fn(embedding_option, embedding_kwargs):
    """
    Returns a function that, when called, applies
    the chosen embedding to the data.
    """
    embedding_kwargs_copy = dict(embedding_kwargs)
    transform_fn = embedding_kwargs_copy.pop("transform_fn", None)

    if embedding_option == "angle":
        return partial(angle_embedding, transform_fn=transform_fn, **embedding_kwargs_copy)
    elif embedding_option == "multiangle":
        return partial(multiple_rotation_embedding, transform_fn=transform_fn, **embedding_kwargs_copy)
    elif embedding_option == "amplitude":
        return partial(amplitude_embedding, **embedding_kwargs_copy)
    elif embedding_option == "basis":
        return partial(basis_embedding, **embedding_kwargs_copy)
    
    # For advnced users: If a new embedding funcion is created, you can add another elif to have it as a built-in option

    else:
        if callable(embedding_option):
            return embedding_option
        raise ValueError(f"Unknown embedding option: {embedding_option}")


def angle_embedding(data, wires, rotations=None, indices=None, transform_fn=None):
    """
    One-to-one angle embedding (one feature per qubit) with
    optional *indices* to pick which features are used when
    len(data) > len(wires).

    Args:
        data (tensor_like): shape (..., n_features) or (n_features,)
        wires (Sequence[int]): target qubits
        rotations (Sequence[str] | None): rotation gate(s); default ["X"]
        indices (Sequence[int] | None): which data indices to embed; Max length must equal len(wires).
        transform_fn (callable | None): optional classical preprocessing
    """
    if transform_fn is not None:
        data = transform_fn(data)     
                           
    if rotations is None:
        rotations = ["X"]

    # Determine how many features there are in the last dimension
    n_wires = len(wires)
    n_data  = data.shape[-1] if hasattr(data, "shape") else len(data)

    # User‑controlled feature selection
    if indices is not None:
        if len(indices) != n_wires:
            raise ValueError(
                f"indices length ({len(indices)}) must equal number of wires ({n_wires})"
            )
        if max(indices) >= n_data:
            raise ValueError("An index in 'indices' is out of range for the provided data")
        data = data[..., indices]                                  
        n_data = n_wires

    elif n_data > n_wires:
        data = data[..., :n_wires] # keep leading features

    # Cyclical repeat when there are fewer features than qubits
    if n_data < n_wires:
        idx = qml.math.arange(n_wires) % n_data                        
        data = data[..., idx]

    # Apply chosen rotation(s)
    for rot in rotations:
        qml.AngleEmbedding(features=data, wires=wires, rotation=rot)  


def multiple_rotation_embedding(data, wires, rotations=("Z", "Y", "Z"), indices=None, transform_fn=None):
    """
    Custom angle embedding that handles batched inputs.
    
    For each rotation, it extracts the corresponding feature from each sample
    in the batch and replicates it to all wires.
    
    For example, if data is a batch with shape (batch_size, 4) and 
    rotations = ["X", "Y", "Z"], then for the first rotation it extracts
    the first column and replicates each value to create a vector of length len(wires),
    resulting in a tensor of shape (batch_size, len(wires)). This is then passed to
    qml.AngleEmbedding.
    
    Args:
        data: A tensor (or array-like) of shape (batch_size, n_features) or (n_features,).
              For a batched input, each sample is assumed to be a row.
        wires: A list (or sequence) of qubit wires.
        rotations (list of str): List of rotation gate names.
        indices: List of element indices in the observation tensor that will be embeded. 
        
    Raises:
        ValueError: If the number of features in each sample is not 1 or equal to len(wires)
                    (for non-batched inputs) or if a given rotation index exceeds n_features.
    """
    # Optional preprocessing
    if transform_fn is not None:
        data = transform_fn(data)

    # Determine batched / un‑batched
    batched = qml.math.ndim(data) > 1
    batch_size, n_features = (data.shape[0], data.shape[-1]) if batched else (1, len(data))
    
    
    # Default mapping if user did not specify one
    n_wires = len(wires)
    indices = list(range(len(rotations))) if indices is None else indices
    if len(indices) != len(rotations):
        raise ValueError("indices and rotations must have the same length")

    # For each rotation, we extract the corresponding feature and replicate it.
    for rot, feat_idx in zip(rotations, indices):
        if feat_idx >= n_features:
            raise ValueError(f"Feature index {feat_idx} out of range for {n_features} feature input")

        column = data[:, feat_idx] if batched else data[feat_idx]          # (batch,) or scalar
        replicated = qml.math.stack([column] * n_wires, axis=-1)               # (batch, wires) or (wires,)
        qml.AngleEmbedding(features=replicated, wires=wires, rotation=rot)


def amplitude_embedding(data, wires):
    """
    Expects data to be length 2**(n_qubits).
    We rely on qml.AmplitudeEmbedding to do the normalization.
    """
    qml.AmplitudeEmbedding(features=data, wires=wires, normalize=True)


def basis_embedding(inputs, wires):
    """
    Encodes a discrete state into a basis state on len(wires) qubits.

    PennyLane's BasisEmbedding supports either:
      - an integer scalar s  (=> prepares |bin(s)⟩ on the wires)
      - a 1-D array of length len(wires), each entry 0 or 1
    """

    # Convert torch.Tensor / np.ndarray to numpy array
    x = np.array(inputs)

    # Flatten any extra dims
    x = x.flatten()

    # Choose integer vs. list form
    if x.size == 1:
        # Single discrete value
        feature = int(x[0])
    elif x.ndim == 1 and x.size == len(wires):
        # Already a binary vector of correct length
        feature = x.astype(int).tolist()
    else:
        raise ValueError(
            f"basis_embedding got inputs of shape {inputs.shape}, "
            f"which flattens to {x.shape}. Expected size 1 or length {len(wires)}."
        )

    # Finally call PennyLane’s template
    qml.BasisEmbedding(features=feature, wires=wires)