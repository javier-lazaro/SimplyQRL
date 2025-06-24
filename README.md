# SimplyQRL
![Project Status: Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow)

*A simple Quantum Reinforcement Learning library*

## Overview

SimplyQRL is a Python library designed to support research and experimentation in **Quantum Reinforcement Learning (QRL)**, with a special focus on **hybrid quantum-classical agents** based on Parameterized Quantum Circuits (PQCs).  

It integrates seamlessly with **Gymnasium** environments, ensuring compatibility with widely used reinforcement learning benchmarks. The quantum components are implemented using **PennyLane**, a leading Python library for differentiable quantum programming, enabling efficient simulation and optimization of quantum circuits alongside classical machine learning tools.  

SimplyQRL provides a modular and extensible framework for comparing classical deep reinforcement learning agents with quantum-enhanced counterparts under standardized and controlled conditions.

## Purpose

The main goal of SimplyQRL is to offer a controlled and configurable benchmarking environment for evaluating the impact of quantum techniques on reinforcement learning performance. It isolates key architectural elements such as:

* **Embedding strategies** for mapping classical observations into quantum circuits
* **Circuit (Ansatz) design** with customizable entanglement patterns
* **Inference layers** connecting quantum outputs to classical decisions

By doing so, it helps researchers understand which quantum techniques provide meaningful advantages and under what conditions.

## Main Features

✅ Classical RL baselines using adapted versions of PPO and DQN (from CleanRL)  
✅ Hybrid quantum-classical agents built on PennyLane  
✅ Modular quantum layer builders (basic, Hsiao, Skolik circuits)  
✅ Support for advanced techniques like **Data Reuploading** and **Output Reuse**  
✅ Flexible embedding options (angle, multi-angle, amplitude, basis)  
✅ Classical preprocessing transformations for observations (e.g., normalization, arctangent scaling)  
✅ Logging, saving, and loading of agents for reproducible experiments  
✅ Idealized quantum simulations using the "lightning.qubit" device (no hardware noise)

## How It Works

SimplyQRL integrates into reinforcement learning pipelines by replacing classical neural network components with quantum-enhanced alternatives. Specifically:

* **Classical agents** use standard multilayer perceptrons (MLPs) as actor-critic or Q-networks.
* **Hybrid agents** swap in PQCs, where classical data is embedded onto quantum states, processed through parameterized layers, and measured to produce outputs.

The library builds on CleanRL's robust algorithm implementations and relies on **Gymnasium** environments for interaction, ensuring that comparisons between classical and quantum agents are fair, systematic, and compatible with common RL benchmarks.

## Supported Algorithms

* **Proximal Policy Optimization (PPO)** → Suitable for discrete action spaces and actor-critic architectures.
* **Deep Q-Network (DQN)** → Suitable for discrete action spaces with value-based learning.

## Repository Structure

* `agents.py`: Defines both classical and hybrid agent architectures.
* `buffers.py`: Provides replay buffers for experience sampling (used in DQN).
* `dqn.py`: Implements the DQN trainer adapted for hybrid agents.
* `ppo.py`: Implements the PPO trainer adapted for hybrid agents.
* `qlayers.py`: Provides quantum circuit layer builders.
* `embeddings.py`: Contains data embedding methods.
* `transformations.py`: Defines preprocessing transforms for observation data.

## Installing and Developing with Poetry

This project uses [Poetry](https://python-poetry.org/docs/) for dependency and environment management. Please follow the official installation guide before proceeding.

Once Poetry is installed, you can set up the development environment as follows:

```bash
# Clone the repository
git clone https://github.com/javier-lazaro/SimplyQRL.git
cd SimplyQRL

# Install dependencies and create a virtual environment
poetry install

# Activate the virtual environment
poetry shell
```

You are now ready to use the library and explore the example scripts in the `examples/` folder.

### Editable Mode (Optional)

To install the library in editable mode for development purposes:

```
poetry install --editable .
```

This allows you to import and test `simplyqrl` from other projects while modifying the source code locally.

### Running Example Scripts

All example scripts are located in the `examples/` folder. You can run them directly with:

```
poetry run python examples/cartpole_dqn.py
```

Make sure to use `poetry run` or activate the Poetry shell beforehand.

## Project Status

The final release of SimplyQRL is **work in progress**. If you are interested, you can request access to a **pre-release version** by opening an Issue on this repository or reaching out via the public email listed in the maintainer's GitHub profile.

**Estimated release date for the initial public version:** Q4 2025.

## Intended Use

This library is intended **for research and educational purposes only**. It is a prototyping tool for testing ideas in QRL, not an industrial-strength framework. Current implementations assume idealized quantum simulators (using PennyLane's `"lightning.qubit"` device) and do not account for noise or hardware-specific constraints.

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it as long as you include the license and copyright notice.

## Citation

If you use SimplyQRL in your research, please cite it as:

```bibtex
@misc{Lazaro2025SimplyQRL,
  author       = {L{\'a}zaro, Javier},
  title        = {SimplyQRL: A Python Library for Hybrid Quantum-Classical Reinforcement Learning},
  year         = {2025},
  url          = {https://github.com/javier-lazaro/SimplyQRL},
  urldate      = {2025-06-03},
  note         = {GitHub repository},
}
```

## Contact

For questions, collaborations, or to request a pre-release version, please open an Issue on this repository or reach out via the public email listed in the maintainer's GitHub profile.

## Acknowledgments

This project draws conceptual inspiration and design patterns from the excellent open-source libraries [CleanRL](https://github.com/vwxyzjn/cleanrl) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). Their contributions to the reinforcement learning community have been invaluable in shaping this work.

