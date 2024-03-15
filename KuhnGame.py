import random

class KuhnGame:

    #we don't care what round we are
    #except that round 4 is "FINISHED"
    JUDGING = 3
    FINISHED = 4

    #display given game state
    def display(self, game_state, prefix = ""):

        #gather some basic info from game state
        pot = game_state["pot"]
        round = game_state["round"]
        cards = game_state["cards"]
        actions = game_state["actions"]

        #print that info out
        print(prefix + "Round {}, Pot: {}, Cards: {}, Actions: {}".format(round,pot,cards,actions))
        #print("===")
        #print(str(game_state))
        #print("===")

    #step a round
    def step(self, game_state):
        #extract cards from game state
        cards = game_state["cards"]
        #players = game_state["players"]

        #trigger round 1
        if game_state["round"] == 0:

            #player 1 always starts
            game_state["history"].append (self.players[0].declare_action(cards[0],["B","C"], game_state))

            #reduce stack of player 1 appropriately, and increase pot as well
            game_state["actions"] += game_state["history"][-1][0]
            game_state["stacks"][0] -= game_state["history"][-1][1]
            game_state["pot"] += game_state["history"][-1][1]

            #move to next round and return game state
            game_state["round"] = 1
            game_state["current_player"] = 1
            return game_state

        #trigger round 2
        if game_state["round"] == 1:

            #player 2 always follows (there is no folding)
            game_state["history"].append(self.players[1].declare_action(cards[1],["B","C"], game_state))

            #reduce stack of player 1 appropriately
            game_state["actions"] += game_state["history"][-1][0]
            game_state["stacks"][1] -= game_state["history"][-1][1]
            game_state["pot"] += game_state["history"][-1][1]

            #move to next round and return game state
            #if player 1 checked and player 2 bet, play returns to player 1
            if game_state["history"][-1][0] == "B" and game_state["history"][-2][0] == "C":
                game_state["round"] = 2
                game_state["current_player"] = 0
            else:
                #skip to last round, there is no round 3
                game_state["round"] = 3
                game_state["current_player"] = 2 #2 indicates none of the players

            return game_state

        #trigger round 3
        if game_state["round"] == 2:

            #if player 1 checked and player 2 bet, play returns to player 1
            if game_state["history"][-1][0] == "B" and game_state["history"][-2][0] == "C":
                #act on player 1
                game_state["history"].append (self.players[0].declare_action(cards[0],["B","C"], game_state))

                #reduce stack of player 1 appropriately
                game_state["actions"] += game_state["history"][-1][0]
                game_state["stacks"][0] -= game_state["history"][-1][1]
                game_state["pot"] += game_state["history"][-1][1]

            #move to next round and return game state
            game_state["round"] = 3
            return game_state
        
        #round 4 is the "judging" round where we pick a winner
        if game_state["round"] == 3:

            #who won?
            winner = -1
            if game_state["stacks"][0] == game_state["stacks"][1]:
                #both players matched bet, the best card wins
                if cards[0] == "K" or cards[1] == "J":
                    winner = 0
                else:
                    winner = 1
            else:
                #whoever bet the most wins
                if game_state["stacks"][0] < game_state["stacks"][1]:
                    #player 1 bet more so player 1 wins
                    winner = 0
                else:
                    #player 2 bet more so player 2 wins
                    winner = 1

            #award the pot to the winners
            game_state["stacks"][winner] += game_state["pot"]
            game_state["pot"] = 0
            game_state["payoff"] = game_state["stacks"][winner] - 2 #we started with 2 chips
            game_state["round"] = KuhnGame.FINISHED
            game_state["winner"] = winner
            game_state["winner_name"] = self.players[winner].name
            return game_state

        #all other rounds, just return game state
        return game_state

    #reset the game state
    def reset(self, players):
        #get random cards for each of our players
        game_state = {}
        kuhn_cards = ['J', 'Q', 'K']
        game_state["cards"] = random.sample(kuhn_cards, 2)
        game_state["history"] = []
        game_state["round"] = 0

        #the player names don't necessarily align to index of players, so track that in game state
        self.players = [players[0], players[1]]
        game_state["player_seats"] = {}
        game_state["player_seats"][players[0].name] = 0
        game_state["player_seats"][players[1].name] = 1

        #setup stacks and pot - assume both players have 2 stack and each ante 1
        game_state["stacks"] = [1,1]
        game_state["pot"] = 2
        game_state["current_player"] = 0
        game_state["actions"] = ""

        #return that game state
        return game_state

    #step round -> moves to next round where current player is the same as the original player
    #or the game is over
    def step_round(self, game_state):

        #get current player
        current_player = game_state["current_player"]

        #step until the current player returns or the game is over
        game_state = self.step(game_state)
        while current_player != game_state["current_player"] and game_state["round"] != KuhnGame.FINISHED:
            game_state = self.step(game_state)

        #if we are on the judging round, perform it now
        if game_state["round"] == KuhnGame.JUDGING:
            game_state = self.step(game_state)

        #we have stepped the round until we returned to the current player
        return game_state

    #play a single game of kuhn poker
    def play(self, players):

        #get a new game state
        game_state = self.reset(players)

        #trigger all 4 rounds, don't worry if some rounds are not applicable
        game_state = self.step(game_state)
        game_state = self.step(game_state)
        game_state = self.step(game_state)
        game_state = self.step(game_state)

        #call game end message for boht players
        winner = game_state["winner"]
        for p in players:
            p.receive_round_result_message(winner, game_state)

        #return final game state if anyone out there cares
        return game_state

