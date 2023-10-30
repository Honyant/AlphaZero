#from multiprocessing import Pool
#from alphazero.game import Game
from alphazero.gamecontainer import GameContainer
from alphazero.model import AlphaZeroNet
import torch
import numpy as np
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#test the game functions:
net = torch.jit.script(AlphaZeroNet(board_area=6*7, num_actions=7, input_depth=8))
net.to(device)

#test the model:
net(torch.from_numpy(np.zeros((1, 8, 6, 7))).float().to(device))

def testGameplay():
    gc = GameContainer()
    
    while not gc.game.terminal():

        # print(gc.game.board)
        # print("Player", gc.game.player, "to play.")
        # print("Possible moves:", gc.get_moves())
        model_input = torch.unsqueeze(torch.from_numpy(gc.get_state()*gc.game.get_player()).float(), 0)
        action_probs, value = net(model_input.to(device))
        action_probs = action_probs.squeeze().detach().cpu().numpy()
        value = value.squeeze().detach().cpu().numpy()

        #random:
        #action_probs = np.random.rand(7)
        #value = np.random.rand(1)
        possible_moves = gc.get_moves()

        #remove illegal moves by setting their probability to 0
        probs = remove_illegal_moves(action_probs, possible_moves)
        
        move = np.random.choice(np.arange(7), p=probs)
        gc.make_move(move)
        # print("Action probabilities:", action_probs)
        # print("Value:", value)
    # print(gc.game.board)
    # if gc.game.get_winner() == 0:
    #     print("Draw!")
    # else:
    #     print("Player", gc.game.get_winner(), "wins!")

def make_move(move: np.array, gc: GameContainer):# on the cpu side, we will parallelize this function, then send the resulting boards back to the gpu
    move = move.squeeze()
    if gc.game.terminal():
        return None, gc
    possible_moves = gc.get_moves()
    probs = remove_illegal_moves(move, possible_moves)
    move = np.random.choice(np.arange(7), p=probs)
    gc.make_move(move)
    return (gc.game.get_winner(), gc)

def remove_illegal_moves(action_probs, possible_moves):
    probs = np.zeros(7)
    for move in possible_moves:
        probs[move] = 1
    probs = probs * action_probs
    probs = probs / sum(probs)
    return probs

def testParallelGameplay(num_processes, num_games=2):
    #create a game container for each process
    def all_terminal(gc_list):
                for gc in gc_list:
                    if not gc.game.terminal():
                        return False
                return True
    
    games_played=0
    wins = {1:0, -1:0, 0:0}
    
    #with Pool(num_processes) as pool:
    start = time.time()
    

    for _ in range(num_games):
        gc_list = [GameContainer() for _ in range(num_processes)]
        
        while not all_terminal(gc_list):
            states = [gc.get_state() for gc in gc_list]
            players = [gc.game.get_player() for gc in gc_list]
            batched_states = torch.from_numpy(np.array(states)*np.array(players)[:, np.newaxis, np.newaxis, np.newaxis]).float()

            action_probs, value = net(batched_states.to(device))
            action_probs = action_probs.detach().cpu().numpy()
            value = value.detach().cpu().numpy()

            action_probs = np.array_split(action_probs, num_processes)
            value = np.array_split(value, num_processes)
            
            #states = pool.starmap(make_move, zip(action_probs, gc_list))
            states = [(make_move(action_probs[i], gc_list[i])) for i in range(num_processes)]
            
            #unpack the states and boards and update the boards:
            gc_list = [gc for _, gc in states]
            states = [state for state, _ in states]
            for i in range(num_processes):
                if states[i] != None:
                    wins[states[i]] += 1

        games_played += 1
        print("Batch", games_played, "finished.")
    print(wins)
    end = time.time()
    return end-start
# for i in range(20):
#     testGameplay()

start = time.time()
num_workers = 256
num_runs = 10
total_time = testParallelGameplay(num_workers, num_runs)
# start = time.time()
# for i in range(num_runs*num_workers):
#     testGameplay()
# end = time.time()
# total_time = end-start-startup_time
print("Games played:", num_runs*num_workers)
print("Workers:", num_workers)
print("Time:", total_time)
print("Seconds per game:", total_time/num_runs/num_workers)
print("Games per second:", 1/(total_time/num_runs/num_workers))