import random
import copy
import numpy as np
from KuhnPlayer import KuhnPlayer 
from KuhnGame import KuhnGame
from Regrets import InformationSet, RegretManager, StrategyManager
from GameAbstractor import KuhnAbstractor

REGRETBOT_DEBUG = False
REGRETBOT_ARGMAX = True #if true we use np.argmax to select a strategy, otherwise we use weighted random probability
REGRETBOT_ALWAYSP1 = False

class RegretBot(KuhnPlayer):

    #initialize the player
    def __init__(self, name, regret_man, abstractor):
        #init super
        super().__init__(name)

        #we are not training, but we do need a regret manager
        self.training = False
        self.regret_man = regret_man
        self.abstractor = abstractor

    #start/stop training
    def set_training(training):
        self.training = training

    #iterate the game state tree
    def iterate_action_tree(self, game_state, actions, reach_probability: float, depth):

        #round 4 means we are finished
        if self.abstractor.round_finished(game_state):
            #return our final stack minus our initial stack (always 2 for kuhn poker)
            payoff = self.abstractor.get_payoff(self, game_state)
            if REGRETBOT_DEBUG: print("   " * depth + ": Payoff = " + str(payoff))
            return payoff

        #get information set for given game state
        info_set = self.regret_man.get_information_set(game_state, self.puppet)

        #get our current strategy from the information set
        strategy = info_set.get_strategy(reach_probability, depth)

        #create a strategy manager for that strategy, to help compute regrets, etc
        #the strategy manager needs to know the # of action sets in order to compute regrets
        strat_man = StrategyManager(strategy, self.regret_man.game_abstractor.action_sets())

        #try each valid action
        for c in range(len(actions)):

            #compute new reach probability after this action
            action_probability = strategy[c]
            new_reach_probability = reach_probability * action_probability

            #make a deep copy of the game state but restore references to player objects
            working_state = copy.deepcopy(game_state)

            #set the puppet action
            #and step the game state forward once
            if REGRETBOT_DEBUG: print("   " * depth + "Testing Action: " + actions[c])
            self.puppet.update_strategy([actions[c]])

            #step the round
            working_state = self.abstractor.step_round(self.emulator,working_state, self.abstractor.abstract_action(actions[c]))

            #if debugging, print out debug info of the game from the abstractor
            if REGRETBOT_DEBUG: self.abstractor.display(working_state,"   " * depth)

            #now, iterate all possible next actions
            #note: that valid actions for all rounds are always B, C (bet or call/fold)
            utility = self.iterate_action_tree(working_state, actions, new_reach_probability, depth + 1)

            #record utility in strategy manager
            strat_man.set_counterfactual_value(c,utility)

        #Value of the current game state is just counterfactual values weighted by action probabilities
        utility = strat_man.get_regret()

        #let the strat manager update the info set based on current reach probability
        strat_man.update_regrets( info_set , reach_probability )

        #return something?
        return utility

    #declare action
    def declare_action(self, card, valid_actions, game_state):

        #get our info set and strategy from regret manager
        info_set = self.regret_man.get_information_set(game_state, self)
        strat = info_set.get_average_strategy()

        #get the action from strategy
        #action_index = np.argmax(strat)
        if REGRETBOT_ARGMAX:
            action_index = np.argmax(strat)
        else:
            if strat[0] == strat[1]:
                action_index = random.choice([0,1])
            else:
                action_value = random.choices(strat, weights=strat, k=1)[0]
                action_index = 0
                if strat[1] == action_value: action_index = 1
        

        #get a random action from our bet strategy and return it
        return self.abstractor.abstract_action(valid_actions[action_index])

    #train the model for a specified # of iterations
    def train(self,iterations, game):
        #alert that we are training
        print("training model for {} iterations...".format(iterations))

        #setup our opponent and our puppet
        #and a game for emulation
        self.puppet = RegretPuppet(self.name)
        self.opponent = RegretBot(self.name+1,self.regret_man, self.abstractor) #play against an exact copy of ourselves with same regret engine
        self.emulator = game

        #run through iterations of self play
        players = [self.puppet, self.opponent]
        for i in range(iterations):
            #reset the game state
            game_state = self.abstractor.reset_game(self.emulator,players)

            #step the game state until our turn
            self.abstractor.step_to_player(self.emulator, self, game_state)

            #now iterate the game with our game state
            utility = self.iterate_action_tree(game_state,self.abstractor.game_actions(),1,0)
            if REGRETBOT_DEBUG: 
                self.abstractor.display(game_state)
                print("---")
                self.regret_man.print_regrets()
                input()


            #reverse player order to train the other way
            if not REGRETBOT_ALWAYSP1: players.reverse()

        print("---")
        self.regret_man.print_regrets()


class RegretPuppet(KuhnPlayer):

    #initialize the player
    def __init__(self, name):
        #init super
        super().__init__(name)

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

