import numpy as np
import random
import sys
import time

from KuhnPlayer import KuhnPlayer 
from KuhnHuman import KuhnHuman 
from RegretBot import RegretBot
from KuhnGame import KuhnGame
from Regrets import RegretManager
from GameAbstractor import KuhnAbstractor

if __name__ == "__main__":

    #fix printing issues
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    #create a game abstractor for kuhn game state
    ga = KuhnAbstractor()

    #quit if we don't have at least 1 argument
    args = sys.argv
    if len(args) < 3:
        print("You must specify at least 2 arguments: T file1 # desc || P file1 file2 # || C file1 file2 || U file1 desc:::")
        args = input().split()
        args.insert(0,"")

    #if we are training
    if args[1] == "T":
        #training iterations is second arg
        num_iterations = int(args[3])

        #train for that amount and quit
        print("Welcome to Kuhn Poker")

        #create a new bot
        reg = RegretManager(ga)
        bot = RegretBot(2,reg, ga)

        #train the bot, track training time
        start_time = time.time()
        bot.train(num_iterations, KuhnGame())
        end_time = time.time()
        print("Training took {:04.2f} minutes, {:04.0f} iterations per second".format((end_time-start_time)/60, num_iterations / (end_time-start_time) ))

        bot.regret_man.name = args[4]
        bot.regret_man.save(args[2])
    elif args[1] == "C":

        #load two regret files
        r1 = RegretManager(ga,args[2])
        r2 = RegretManager(ga,args[3])

        #compare them
        print("Comparison of {} and {}".format(r1.name,r2.name))
        print(" {} | {}".format(r1.name,r2.name))
        r1.compare_regrets(r2)

    elif args[1] == "V":

        #load a regret file
        r1 = RegretManager(ga,args[2])

        #view it 
        print("Contents of {}".format(args[2]))
        r1.print_regrets(True)

    elif args[1] == "E":

        #load a regret file
        r1 = RegretManager(ga,args[2])

        #view it 
        print("Training Summary for {}".format(args[2]))
        print("REIMPLEMENT!")

    elif args[1] == "U":

        #update name of strategy
        r = RegretManager(ga,args[2])
        r.name = args[3]
        r.save(args[2])

    else:

        #create two bots
        players = [
            RegretBot(1, RegretManager(ga,args[2]), ga),
            RegretBot(2, RegretManager(ga,args[3]), ga)
        ]

        #compare them
        print("Tournament of {} and {}".format(players[0].regret_man.name,players[1].regret_man.name))
        print(" {} | {}".format(players[0].regret_man.name,players[1].regret_man.name))

        #setup the game
        game = KuhnGame()
        num_games = int(args[4])
        total_winnings = [0,0,0]
        tourney_wins = [0,0]
        wins = [0,0]
        tourney_games = 1000
        for i in range(int(num_games)):

            #reset this tourney
            winnings = [0,0]

            for i2 in range(tourney_games):

                #play the game and get final state
                game_state = game.play(players)

                #add to each winnings
                winnings[0] += game_state["stacks"][game_state["player_seats"][1]] - 2
                winnings[1] += game_state["stacks"][game_state["player_seats"][2]] - 2
                total_winnings[2] += game_state["payoff"]

                #reverse seats
                players.reverse()

                #who won this game
                wins[game_state["winner_name"]-1] += 1

            #whoever one the most winnings - count
            if winnings[0] > winnings[1]:
                tourney_wins[0] += 1
            else:
                tourney_wins[1] +=1

            #add to total winnings
            total_winnings[0] += winnings[0]
            total_winnings[1] += winnings[1]

        print(players[0].regret_man.name + " Won " + str(tourney_wins[0]) + " Tournaments, " + str(wins[0]) + " Games, Total Winnings: " + str(total_winnings[0]))
        print(players[1].regret_man.name + " Won " + str(tourney_wins[1]) + " Tournaments, " + str(wins[1]) + " Games, Total Winnings: " + str(total_winnings[1]))
        print("Nash ~= {:04.2f}% by payoff".format(total_winnings[0] / total_winnings[2] * 100))
        