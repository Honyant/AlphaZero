import numpy as np
def remove_illegal_moves(action_probs, possible_moves):
    probs = np.zeros(7)
    for move in possible_moves:
        probs[move] = 1
    probs = probs * action_probs
    probs = probs / sum(probs)
    return probs