import random

class KuhnPlayer:

    #initialize the player
    def __init__(self, name, bet_strategy = None):
        self.name = name
        self.bet_strategy = bet_strategy

    #update betting strategy (useful for training)
    def update_strategy(self,bet_strategy):
        self.bet_strategy = bet_strategy

    #declare action
    def declare_action(self, card, valid_actions, game_state):
        #get a random action from our bet strategy and return it

        action = random.choice(self.bet_strategy)
        value = 0
        if action == "B": value = 1
        return action, value

    #receive round result message
    def receive_round_result_message(self, winner, game_state):
        pass

