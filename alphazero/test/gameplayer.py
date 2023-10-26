from alphazero.game import Game

#play game with input:
def play_game(game: Game):
    while not game.terminal():
        print(game.board)
        print("Player", game.player, "to play.")
        move = int(input("Enter a move: "))
        game.apply(move)
    print(game.board)
    if game.get_winner() == 0:
        print("Draw!")
    else:
        print("Player", game.get_winner(), "wins!")

play_game(Game())