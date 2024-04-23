import torch
from flask import Flask, jsonify, render_template
from mcts import Node, mcts_search
import numpy as np
from game import step

from network import AlphaZeroNet

app = Flask(__name__)


def traverse(node, board, cur_player=1):
    if node is None:
        return None

    # Generate the board states for the children
    children = []
    for i, child in enumerate(node.children):
        if child is not None:
            # Get the action from the edges array
            action = node.edges[i, 4].astype(int)
            # Make a copy of the board
            board_copy = np.copy(board)
            # Apply the action to the board copy
            _, _ = step(board_copy, action)
            board_copy *= -1
            # Recursively traverse the child
            children.append(traverse(child, board_copy, -cur_player))

    return {
        'board': board.tolist(),
        'children': children,
        'visits': [] if node.leaf else node.edges[:, 0].tolist(),
        'priors': [] if node.leaf else node.edges[:, 2].tolist(),
        'q_values': [] if node.leaf else node.edges[:, 3].tolist(),
        'actions': [] if node.leaf else node.edges[:, 4].tolist(),
        'cur_player': cur_player,
        'leaf': node.leaf
    }


@app.route('/tree')
def tree():
    # Generate an MCTS Search Tree starting from a blank board with 200 iterations
    mcts_hyperparams = {
        'iterations': 1000,
        'c_puct': 1.0,
        'tau': 1,
        'device': torch.device('cpu'),
        'rollouts': 1000,
        # 'mps' if torch.backends.mps.is_available() else
    }

    board = np.array([
        [-0., -0., -0., -0., -0., -0., -0.],
        [-0., -0., -0., 1., -0., -0., -0.],
        [-0., -0., 1., 1., -0., 1., -0.],
        [-0., -0., 1., -1., -0., 1., -0.],
        [-0., -0., -1., 1., -0., -1., -1.],
        [-0., -1., -1., 1., -1., -1., -1.]
    ]).astype(np.int8)

    root = Node(None, None)

    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=2).to(mcts_hyperparams['device'])
    net.load_state_dict(torch.load('model_drawn_sunset_42.pth'))

    mcts_search(board, root, net, mcts_hyperparams, use_model=False)
    return jsonify(traverse(root, board))


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
