
import pennylane as qml
from .embeddings import multiple_rotation_embedding, angle_embedding
from .transformations import ArctanTransformer, CartPoleNormalizationTransformer
from .transformations import FrozenNormalizationTransformer, FrozenBasisToAngleTransformer

def build_basic_qlayer(embedding_fn, n_qubits, n_layers, dev=None):
    """
    Constructs a Torchlayer based on a basic PennyLane QNode that applies a custom embedding followed by a
    parameterized quantum circuit (using BasicEntanglerLayers) and returns the 
    expectation values of Pauli-Z measurements on each qubit.

    Args:
        embedding_fn (callable): A function with signature (inputs, wires) that 
            applies the desired data embedding. This function may also accept
            extra kwargs via partial if needed.
        n_qubits (int): The number of qubits (and wires) in the circuit.
        n_layers (int): The number of layers for the trainable quantum circuit.
        dev (qml.Device, optional): A PennyLane device. If None, defaults to a 
            "lightning.qubit" device.

    Returns:
        qml.qnn.TorchLayer: A TorchLayer wrapping the QNode, with a weight shape 
            of {"weights": (n_layers, n_qubits)}.
    """
    if dev is None:
        dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        # 1. Apply the embedding: this function is responsible for encoding
        #    the inputs onto the specified wires.
        embedding_fn(inputs, wires=range(n_qubits))
        # 2. Apply trainable quantum layers (here using BasicEntanglerLayers).
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        # 3. Measure expectation values of PauliZ on each qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def build_hsiao_qlayer(n_qubits, n_layers, emb_type="multi", emb_indices=None, transform_fn=None, ent=False, dev=None):
    """
    Constructs a Torchlayer based on the qcircuit in Hsiao's paper. 
    It applies a special multiple angle into 1 qubit encoding. that applies a custom embedding followed by a
    parameterized quantum circuit (using BasicEntanglerLayers) and returns the 
    expectation values of Pauli-Z measurements on each qubit.

    Returns:
        qml.qnn.TorchLayer: A TorchLayer wrapping the QNode, with a weight shape 
            of {"weights": (n_layers, n_qubits)}.
    """
    if dev is None:
        dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):

        # 1. Apply a perfect superopsition state via Hadamard gates
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        
        for layer in range(n_layers):
            # 2. Apply the embedding: Multiple rotations over same qubit or single rotation per qubit
            if emb_type == "multi":
                multiple_rotation_embedding(inputs, wires=range(n_qubits), rotations=['Z','Y','Z'],
                                             indices=emb_indices, transform_fn=transform_fn)
            elif emb_type == "single": 
                angle_embedding(inputs, wires=range(n_qubits), rotations=['Z','Y'], 
                                indices=emb_indices, transform_fn=transform_fn)

            # 3.  Loop over each qubit and apply the RX gate using the corresponding parameter
            for i, wire in enumerate(range(n_qubits)):
                qml.RX(weights[layer, i], wires=wire)

             # 3. Add circular entanglement with CNOT gates.
            if ent == True:
                for i in range(n_qubits):
                    next_wire = (i + 1) % n_qubits  # Ensures the last qubit connects to the first.
                    qml.CNOT(wires=[i, next_wire])

        # 4.  Measure expectation values of PauliZ on each qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Allocate a separate parameter for each layer and qubit.
    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def build_skolik_qlayer(n_qubits, n_layers, emb_indices=None, transform_fn=None, ent=False, dev=None):
    """
    Constructs a Torchlayer based on the qcircuit in Skolik's paper. 
    It applies a simple angle encoding with Rx gates. That is followed by Ry, Rz parametesired 
    rotations and circular CZ entanglement and returns the expectation values of Pauli-Z measurements on each qubit.
    Multiple layers can be used using the Data Reuploading technique.

    Args:
        n_qubits (int): The number of qubits (and wires) in the circuit.
        n_layers (int): The number of layers for the trainable quantum circuit.

    Returns:
        qml.qnn.TorchLayer: A TorchLayer wrapping the QNode, with a weight shape 
            of {"weights": (n_layers, n_qubits)}.
    """
    if dev is None:
        dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        
        for layer in range(n_layers):
            # 1. Apply the embedding: Multiple rotations over same qubit.
            angle_embedding(inputs, wires=range(n_qubits), rotations=['X'], 
                            indices=emb_indices, transform_fn=transform_fn) 
                        
            # 2.  Loop over each qubit and apply RY, RZ gate using the corresponding parameters.
            for i, wire in enumerate(range(n_qubits)):
                # Paramter gates: Ry and Rz.
                qml.RY(weights[layer, 0, i], wires=wire)
                qml.RZ(weights[layer, 1, i], wires=wire)

            # 3. Add circular entanglement with CZ gates.
            if ent == True:
                for i in range(n_qubits):
                    next_wire = (i + 1) % n_qubits  # Ensures the last qubit connects to the first.
                    qml.CZ(wires=[i, next_wire])
            
        # 3.  Measure expectation values of PauliZ on each qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Allocate a separate parameter for each layer and qubit.
    weight_shapes = {"weights": (n_layers, 2, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)

def build_dr_qlayer(n_qubits, n_layers, emb_indices=None, transform_fn=None, ent=False, dev=None):
    """
    Constructs a Torchlayer based on the qcircuit in the original Data Reuploading paper. 
    It applies a simple angle encoding with Rx gates. That is followed by Ry, Rz parametesired 
    rotations and circular CZ entanglement and returns the expectation values of Pauli-Z measurements on each qubit.
    Multiple layers can be used using the Data Reuploading technique.

    Args:
        n_qubits (int): The number of qubits (and wires) in the circuit.
        n_layers (int): The number of layers for the trainable quantum circuit.

    Returns:
        qml.qnn.TorchLayer: A TorchLayer wrapping the QNode, with a weight shape 
            of {"weights": (n_layers, n_qubits)}.
    """
    if dev is None:
        dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):

        n_data = inputs.shape[-1] if hasattr(inputs, "shape") else len(inputs)
        
        for layer in range(n_layers):
            # 1. Apply the embedding: Multiple rotations over same qubit or single rotation over each qubit, depending on qubit count.
            if n_qubits < n_data:
                multiple_rotation_embedding(inputs, wires=range(n_qubits), rotations=['Z','Y','Z'], 
                                            indices=emb_indices, transform_fn=transform_fn)
            else:
                angle_embedding(inputs, wires=range(n_qubits), rotations=['Z','Y'], 
                                indices=emb_indices, transform_fn=transform_fn)
            
            # 2.  Loop over each qubit and apply RY, RZ gate using the corresponding parameters.
            for i, wire in enumerate(range(n_qubits)):
                # Paramter gates: Ry and Rz.
                qml.RZ(weights[layer, 0, i], wires=wire)
                qml.RY(weights[layer, 1, i], wires=wire)
                qml.RZ(weights[layer, 2, i], wires=wire)

            # 3. Linear entanglement with CZ (skip if last layer or only 1 qubit)
            if ent and (layer < n_layers - 1) and (n_qubits > 1):
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            
        # 3.  Measure expectation values of PauliZ on each qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Allocate a separate parameter for each layer and qubit.
    weight_shapes = {"weights": (n_layers, 3, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)