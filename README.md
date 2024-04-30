# AlphaZero: A DeepMind Algorithm Replication

## Project Title and Description

This project is aimed at replicating DeepMind's AlphaZero algorithm, specifically for the game of Connect 4. The project uses a combination of Monte Carlo Tree Search (MCTS) and a deep convolutional neural network to train an AI agent capable of playing Connect 4 at a high level.

## Installation and Setup

1. Clone the repository: `git clone https://github.com/Honyant/AlphaZero.git`
2. Navigate to the project directory: `cd AlphaZero`
3. Install the required Python packages: `pip install -r requirements.txt`

## Usage and Examples

To train the AI, run the `train.py` file:

```bash
python train.py
```

To train the AI in parallel, run the `parallel_train.py` file:

```bash
python parallel_train.py
```

To play against the AI, run the `ai_game.py` file:

```bash
python ai_game.py
```

## Acknowledgements and Credits

This project was inspired by DeepMind's AlphaZero algorithm. It uses the PyTorch library for the neural network implementation.