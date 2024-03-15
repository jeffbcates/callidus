#testing
from ai_caller import FishPlayer 

from multiprocessing import shared_memory
from Regrets import InformationSet, RegretManager, StrategyManager
from PokerAbstractor import PokerAbstractor
from util import estimate_prob, progressBar, setup_emulator_players
from TrainingProgress import TrainingProgress

from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.hand_evaluator import HandEvaluator as he
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck
from pypokerengine.utils.game_state_utils import deepcopy_game_state
from pypokerengine.api.game import start_poker, setup_config
import random
import sys
import operator
import pickle
import numpy as np
import time

ABSTRACTED_ACTIONS = 9
REGRETBOT_WEIGHTEDSTRATS = 2 #how many of the available actions will we consider when applying a weighted strategy
REGRETBOT_FOCUSTIME = 7 #how long we give for focused training during live play (measured in seconds)
REGRETBOT_ITERLOOPS = 1 #how many times we loop training a single regret node before moving on
REGRETBOT_STRATEGYTHRESHOLD = 0.01 #we do not continue to train strategies with an average strategy below this
DEBUG_MODE = False
def log(msg):
    if DEBUG_MODE: print("[D]: %s" % msg)

class EmulatorPlayer(BasePokerPlayer):

    #the following is a class function that setups
    #a single trainer in parallel trainer mode
    def paralleltrainer(identity,status_size, status_memory_name, shared_space_name, on_disk, training_timelimit, use_argmax, use_cfr, use_monte, relative_size):

        #get our status memory instance
        status = shared_memory.ShareableList(name=status_memory_name)

        #load our regret manager
        regret_man = RegretManager(PokerAbstractor())
        regret_man.relative_size = relative_size

        #attach monte if we are using it
        if use_monte: regret_man.game_abstractor.open_monte("SHARED_MONTE")
        
        #open on disk?
        if on_disk: regret_man.open(shared_space_name)
        else: regret_man.attach(False)

        #create a new trainer and set identity
        cfr_bot = EmulatorPlayer(regret_man)
        cfr_bot.use_argmax = use_argmax
        cfr_bot.use_cfr = use_cfr
        cfr_bot.use_monte = use_monte
        cfr_bot.regret_man.set_shared_identity(identity)

        #train our bot
        cfr_bot.train(simulations=0,training_timelimit = training_timelimit, verbosity=0, external_status_array = status)


    def __init__(self, regret_man = None):
        super().__init__()
        self.allow_retraining = False
        self.uuid = None
        self.training_timelimit = 0
        self.INTERNAL_ITERS = 0

        #settings for how we declare action
        self.use_argmax = False
        self.use_monte = True
        self.use_cfr = True

        #by default we do not save off untrained states for later review
        self.save_untrained_states = False
        
        #if no regret manager was passed, create a new one
        if regret_man == None:
            #create a new abstractor and regret manager
            self.abstractor = PokerAbstractor()
            self.regret_man = RegretManager(self.abstractor)
        else:
            #use the existing values
            self.regret_man = regret_man
            self.abstractor = regret_man.game_abstractor

        #reset some other values
        self.rounds = 0
        self.verbose = False

        #used for tracing a specific 
        self.trace_path = []
        
    # setup Emulator with passed game information
    def receive_game_start_message(self, game_info):
        pass

    #iterate through all possible actions
    def iterate_action_tree(self, hole_card, game_state, actions, reach_probability, depth, strategy_threshold = REGRETBOT_STRATEGYTHRESHOLD):

        self.INTERNAL_ITERS += 1
        if self.uuid != 'p0':
            print('issue!')

        #if game state is finished, return stack
        if game_state["street"] == Const.Street.FINISHED:
            final_stack = [player for player in game_state['table'].seats.players if player.uuid == self.uuid][0].stack

            #JBC 10/10/20 -> abstract final stack so that hand is a ratio against starting stack
            #so that our starting stack does not influence the value of the win
            final_stack = (final_stack - self.street_start_stack) / self.street_start_stack

            #JBC 10/6/20 -> just return final stack now, since we are not using anything but utility
            #and at the terminal state probability is 100%
            return final_stack, game_state

        #quit if very low reach probability - this will speed up the game
        if reach_probability == 0: 
            #JBC 10/6/20 -> just return a ZERO utility since the probabily of reaching this node is ZERO
            return 0, game_state
    
        #flatten our actions
        round = game_state["round_count"]
        street = game_state["street"]
        table = game_state["table"]
        actions = self.abstractor.flatten_actions(actions, table=table)

        #track action results
        action_states = [None for i in range(len(actions))]
        
        #for tracing:
        path_found = False
        if len(self.trace_path) > 0:
            joined_path = self.abstractor.gen_regret_path(game_state,self)
            if joined_path[:len(self.trace_path)] == self.trace_path[:]: 
                path_found = True
                
        #get information set for given game state
        info_set = self.regret_man.get_information_set(game_state, self)

        #repeat until the number of writes to this infoset is at least our regret loop threshold
        first_loop = True
        while info_set.writes() < REGRETBOT_ITERLOOPS or first_loop:

            #we should at least loop once
            first_loop = False

            #get our current strategy from the information set
            #this will update the strategy sum and then return the newly updated average strategy
            strategy = info_set.get_strategy(reach_probability, depth)

            #create a strategy manager for that strategy, to help compute regrets, etc
            #the strategy manager needs to know the # of action sets in order to compute regrets
            strat_man = StrategyManager(strategy.copy(), self.regret_man.game_abstractor.action_sets())

            #for each of the possible actions
            for c in range(len(actions)):
        
                #if this is a valid action we should iterate it
                #skip node if very low reach probability
                if actions[c].get("valid",True) and strategy[c] > strategy_threshold:

                    #compute new reach probability after this action
                    action_probability = strategy[c]
                    new_reach_probability = reach_probability * action_probability

                    #make a copy of the game state
                    #so we don't mangle it with our testing
                    working_state = deepcopy_game_state(game_state)
        
                    #get action info
                    action = actions[c]
                    action_name = action["action"]
                    action_amount = action["amount"]

                    #step the emulation with our action
                    #self.model.set_action(action_name,action_amount)
                    working_state, _events = self.emulator.step_round(working_state,action_name,action_amount)
            
                    #try to get valid actions (but if round is finished, there won't be any)
                    valid_actions= None
                    if working_state["street"] != Const.Street.FINISHED:
                        valid_actions = self.emulator.generate_possible_actions(working_state)

                    #get all possible next actions from state
                    #and call iterate with those            
                    utility, action_state = self.iterate_action_tree(hole_card,working_state, valid_actions, new_reach_probability, depth + 1, strategy_threshold)
                
                    #save action state
                    action_states[c] = action_state

                    #record utility in strategy manager
                    #JBC: 11/9/20 -> utility should NOT be multiplied by -1 because higher is better
                    strat_man.set_counterfactual_value(c, utility)

                else:

                    #JBC: 10/6/20 -> zero out the strategy of any actions we did not take
                    #what happens when strategy is zero
                    utility = 0
                    strategy[c] = 0

            #JBC: 10/6/20 -> normalize this current strategy (since we previously zeroed out some of the strategies)
            #and update it in strategy manager (this will prevent us from assigning a false value to paths we did not take)
        
            #JBC: 10/9/20 -> do not normalize strategy since the zeroed out strategies could be selected at a later date)
            #strategy = info_set.normalize(strategy)

            #JBC: 11/9/20 -> clear our zeroed strategies so they aren't negatively impacted but DO NOT NORMALIZE
            strat_man.strategy = strategy

            #let the strat manager update the info set based on current reach probability
            strat_man.update_regrets( info_set , reach_probability)

        #now that we have looped enough times - get the average strategy from the infoset
        strategy = info_set.get_average_strategy()

        #if we are tracing - write to our file
        if path_found:
            #open the file for appending
            trace_file = open("regrettrace.txt","a")

            #write to our trace file
            trace_file.write(
                "{}|{}|{}||{}||{}||{}".format(
                    "{}".format(int(time.time())),
                    "-".join([str(t) for t in self.abstractor.unpack_path_tuple(joined_path)]),
                    "{:4.2f}".format(utility),
                    "|".join(["{:4.2f}".format(s) for s in info_set.get_average_strategy()]),
                    "|".join(["{:4.2f}".format(s) for s in info_set.strategy()]),
                    "|".join(["{:4.2f}".format(r) for r in info_set.regrets()]),
                ) + "\n"
            )

            #close the file for writing
            trace_file.close()

        #pick best action -> return that as final state for this path
        #get the action from strategy
        if self.use_argmax:
            #use argmax to pick an action from our strategy
            best_action_index = np.argmax(strategy)
        else:
            #pick a random action from the strategy
            #but weight based on the strategy
            best_action_index = random.choices(range(len(strategy)), weights=strategy, k=1)[0]

        #JBC 10/6/20 -> big change, we should be returning the utility of this entire node-set not the best action found
        #JBC 11/9/20 -> should be returning the strategy managers calculated total regret (not utility becuase that's declared in a loop above)
        return strat_man.get_regret(), action_states[best_action_index]

    #as soon as an epoch produces no new regrets, we need training
    def train(self, simulations, starting_holes = [], initial_state = None, training_timelimit = 0, verbosity = 2, strategy_threshold = REGRETBOT_STRATEGYTHRESHOLD, training_path = None, external_status_array = None):

        #if we have an initial state, match the player count
        #otherwise use the default player count (6)
        if initial_state == None: nb_player = 6
        else: nb_player = len(initial_state["table"].seats.players)

        #start a new emulator
        max_round = 1000
        initial_stack = 10000
        sb_amount = 5
        ante_amount = 0
        emulator = Emulator()
        emulator.set_game_rule(nb_player, max_round, sb_amount, ante_amount)
        self.emulator = emulator

        #calculate reset amount
        reset_amount = (
            self.abstractor.unpack_abstracted_amount(8,bigblind=sb_amount*2) + 
            self.abstractor.unpack_abstracted_amount(9,bigblind=sb_amount*2)
        ) / 2

        #set our uuid if not already set
        if self.uuid == None: self.uuid = "p0"

        #lookup uuids
        if initial_state != None: uuids = [p.uuid for p in initial_state['table'].seats.players]
        else: uuids = []

        #setup other players
        player_info = {}
        players = [self] + [EmulatorPlayer(self.regret_man) for i in range(nb_player-1)]
        player_info = setup_emulator_players(emulator,players,uuids,[initial_stack])

        #make sure all player settings match our settings
        for p in players:
            if p != self:
                p.use_argmax = self.use_argmax
                p.use_cfr = self.use_cfr
                p.use_monte = self.use_monte

        #assign the current players hole cards if we have an initial state
        #this is what keeps our starting hole static while we train with other holes
        #for the other players, and also keep the rest of game state the same
        if initial_state != None:
            player_info[self.uuid]["starting_hole"] = [
                        str(initial_state["table"].seats.players[0].hole_card[0]),
                        str(initial_state["table"].seats.players[0].hole_card[1])
                    ]

        #geneate an initial state for the game unless one was provided
        using_initial_state = True
        if initial_state == None: 
            initial_state = emulator.generate_initial_game_state(player_info)
            using_initial_state = False


        #figure out a good epoch size
        if simulations > 100: epoch_size = int(simulations / 100)
        elif simulations > 0: epoch_size = int(simulations / 10)
        else: epoch_size = 100

        #focused state is when there is a game going on and we need all trainers to focus
        #on a particular game state to quickly train it during live play
        focused_state = None

        #run our simulations
        game_state = None
        starting_hole = 0
        starting_hole_len = len(starting_holes)
        epochs = int(simulations / epoch_size)

        #focusing -> this is an int value that determines if we are just training (0), or training one of other holes (1..2, etc)
        focusing = 0

        #testing - > generate state from training path
        if training_path != None:
            #we are now using the initial state (geneated from path)
            initial_state = self.abstractor.step_emulator_to_path(training_path,initial_state,emulator,players,player_info)
            using_initial_state = True

            #get our genearted path
            check_path = self.abstractor.gen_regret_path(initial_state,self)

            #and compare it to the original requested path if they are off, there is an issue
            if check_path != training_path + [(0,1)]:
                print("Generated Emulator State Does Not Match Training Path!  Aborting Training")
                print("Req: " + str(training_path + [(0,1)]))
                print("Gen: " + str(check_path))
                return

        #setup a new training progress timer
        #the last value of the external status array stores the status size as a helper
        status_length = 0
        if external_status_array != None: status_length = self.regret_man.shared_identity * external_status_array[-1]
        timer = TrainingProgress(external_status_array,status_length)
        timer.start_training(self.regret_man, epochs,epoch_size,training_timelimit)
        
        #start with no epochs completed and not done training
        epoch = 0
        done_training = False

        #train until conditions indicate we are done (either time limit or # of epochs reached)
        while not done_training:

            #this is the epoch we are on
            epoch += 1

            #perform each simulation in the epoch (epoch_size = # of simulations per epoch)
            for c in range(epoch_size):

                #we are starting the next iteration
                self.regret_man.training_iteration += 1

                #if we were passed a list of starting holes, get the next
                #one in the list and assign to player (in this manner you can train specific starts)
                if starting_hole_len > 0: 
                    player_info[self.uuid]["starting_hole"] = [
                        str(starting_holes[starting_hole][0]),
                        str(starting_holes[starting_hole][1])
                    ]
                
                #assume the game does not need to be reset
                game_finished = False
                reset_game = False
                    
                #if we don't have a game state yet, we need to start a new game from initial state
                if game_state == None:
                    game_finished = True
                    reset_game = True

                #start a new game if a winner has been decided
                #or if we are no longer in the game
                #or if there is only 1 player still active (he beat everyone else before the last round)
                if game_state != None:

                    #JBC: 11/12/2020 -> reset stacks of all players here
                    #so that we can better train the various raise scenarios (otherwise they are often unavialable)
                    for p in game_state["table"].seats.players:
                        p.stack = initial_stack

                    #reset round if at last round or # of players is too low (which won't be the case with fixe stacks above)
                    if emulator._is_last_round(game_state,{"max_round":max_round}) or game_state["table"].seats.players[0].stack < reset_amount or game_state["table"].seats.count_active_players() < 2:
                        #the game is finished and we need to reset the game
                        #because the trainee cannot play further
                        game_finished = True
                        reset_game = True

                #we need to start a new round (but dont reset game state)
                if game_state != None:
                    if game_state["street"] == Const.Street.FINISHED: game_finished = True

                #figure out what level of focus we should train
                if focusing and not self.focused_training():

                    #we were focusing, but no longer
                    focusing = 0
                    focused_state = None

                elif self.focused_training() != focusing:

                    #we are now at a greater state of focus
                    focusing = self.focused_training()
                    focused_state = self.focused_state()

                #focusing = self.focused_training()

                #if the game is finished, start a new round or possibly reset the game completely
                if game_finished or focusing:
                    #here we are starting from our initial state
                    #and restoring the trainee's cards and table cards (but not other player cards)
                    if using_initial_state or self.focused_training():

                        #copy from focus state or initial state
                        if focusing:
                            game_state = deepcopy_game_state(focused_state)
                        else:
                            game_state = deepcopy_game_state(initial_state)

                        #save off community cards and player cards (we have to shuffle the deck 
                        community_cards = [str(c) for c in game_state["table"].get_community_card()]
                        player_cards = [str(c) for c in game_state["table"].seats.players[0].hole_card]
                        player_info[self.uuid]["starting_hole"] = player_cards
                        
                        #re-assign hole cards
                        game_state = self.abstractor.assign_hole_cards(game_state, player_info, community_cards)

                        #add back community cards
                        game_state["table"]._community_card = gen_cards(community_cards)

                    else:

                        #here we do not need to reset from initial state
                        #we are just starting the next round
                        if not reset_game:

                            #start the next round, but note - the next round could immediately end
                            #for various reasons (last opponent is out, etc)
                            game_state, events = emulator.start_new_round(game_state)

                            #if we are already at FINISHED state, we should start over completely
                            if game_state["street"] == Const.Street.FINISHED: reset_game = True

                        #now, either we already knew we had to reset game state from initial
                        #or we just tried to start a new round and game was immediately finished
                        #so, now we do need to reset the game from initial state
                        if reset_game: 

                            #JBC 11/11/20 -> since we are reseting game state every round now need to move dealer manually
                            #if initial_state["table"].dealer_btn == nb_player - 1:
                            #    initial_state["table"].dealer_btn = 0
                            #else:
                            #    initial_state["table"].dealer_btn += 1

                            game_state, events = emulator.start_new_round(initial_state)

                        #assign hole carfds - this will also restore and shuffle the deck for us
                        #since we can't assign hole cards without doing that
                        if starting_hole_len > 0: game_state = self.abstractor.assign_hole_cards(game_state, player_info)

                #get our starting path
                starting_path = self.abstractor.gen_regret_path(game_state,self)

                #JBC - lock mechanism doesn't do anyhting now that we are in-memory,
                #just notes which hole we are training (maybe useful for making sure no trainers are hung with a console command)
                lock_aquired = self.regret_man.lock(starting_path[:1])

                #step to our position -> we are always player ZERO
                game_state, messages = emulator.step_to_player_turn(game_state, 0)                

                #it is possible we never reach a chance to play (maybe all the players folded)
                #so before we iterate the action tree, let's check that need to
                if game_state["street"] != Const.Street.FINISHED:

                    #get the actual current path
                    current_path = self.abstractor.gen_regret_path(game_state,self)

                    #geneate valid actions for current player
                    valid_actions = emulator.generate_possible_actions(game_state)

                    #iterate the action tree, return the best action state and simply use that
                    #so we don't have to step back through the game tree to apply the best action we figured out
                    self.street_start_stack = initial_stack
                    hole_card = [str(h) for h in game_state["table"].seats.players[0].hole_card]
                    
                    #iterate action tree and return final game state
                    self.INTERNAL_ITERS = 0
                    utility, game_state = self.iterate_action_tree(hole_card,game_state,valid_actions,1,0, strategy_threshold)
                else:

                    #current path needs to use starting path, since we do not have a valid current path
                    current_path = starting_path

                #if we don't have an initial game state or we do have a shared identity
                #we should save every time (2) for other trainers to access and (1) in case we die
                if self.regret_man.shared_identity != None or not using_initial_state:
                    #unlock path, we are done using it
                    self.regret_man.unlock(starting_path[:1])

                #move to next starting hole for simulation
                if starting_hole_len > 0: 
                    if starting_hole == starting_hole_len-1: starting_hole = 0
                    else: starting_hole += 1

                #within each epoch report on the epoch progress
                new_regrets = timer.report_epoch_progress(self.regret_man, epoch, c, current_path, verbosity, focusing, self.INTERNAL_ITERS )

            #at the end of each epoch, report epoch results
            (new_regrets, done_training) = timer.report_epoch_results(self.regret_man, epoch, verbosity)

        #at the end of training, report training results and return to caller
        return timer.report_training_results(self.regret_man,verbosity)

    #focus current training on a single game state
    def focus_training(self, game_state, sleeptime = 0):

        #if we are training or focusing
        focus = self.regret_man.regret_locks[0]
        if focus[:5] in ["TRAIN","FOCUS"]:

            #if training, intensity is 1
            if focus == "TRAIN": intensity = 1
            else: intensity = int(focus[5:]) + 1

            #pickle our current state
            with open("gamestate.FOCUS{}".format(intensity),"wb") as f:
                pickle.dump(game_state,f)

            #update the focus value in regret locks - this will trigger trainers to reload
            self.regret_man.regret_locks[0] = "FOCUS{}".format(intensity)

            #should we also sleep, and then reduce intensity
            if sleeptime > 0:

                #wait the specified amount of time
                time.sleep(sleeptime)

                #reduce focus intensity to what it was
                self.regret_man.regret_locks[0] = focus

        else:

            #let the user know what we are doing
            print("Self Training...")

            #here, we are not training, so just run a few simulations
            self.train(25, initial_state = game_state, verbosity = 1, strategy_threshold=0.05)

    #unfocus training - when we are no longer focusing on a specific hole
    def unfocus_training(self):

        #go back to training
        self.regret_man.regret_locks[0] = "TRAIN"
    
    #get focused state - load from pickled file
    def focused_state(self):
        #read game state and return it
        with open("gamestate." + self.regret_man.regret_locks[0],"rb") as f:
            state = pickle.load(f)
        return state

    #are we in focused training?
    def focused_training(self):
        #this allows us to have multiple levels of "focus"
        focus = self.regret_man.regret_locks[0][5:6]
        if focus == "1": return 1
        elif focus == "2": return 2
        else: return 0

    #return our action
    def declare_action(self, valid_actions, hole_card, round_state):
        #get current pot from round state
        pot = round_state["pot"]["main"]["amount"]

        #flatten our actions (convert "raise" to "raise0","raise1","raise2", etc)
        valid_actions = self.abstractor.flatten_actions(valid_actions, pot=pot, bb=round_state["small_blind_amount"] * 2)

        #our simulation results
        action_names = [valid_actions[i]["action"] for i in range(len(valid_actions))]
        street = round_state["street"]
        round = round_state["round_count"]
        community_card = round_state["community_card"]
        
        #setup game state
        game_state = self._setup_game_state(round_state, hole_card)
        
        #get our strategy from the information set
        if self.regret_man.symm_tree != None:
            info_set = self.regret_man.get_information_set(game_state, self, False)
            strat = info_set.get_average_strategy()

            #if the strategy is default, and we allow training
            #we need to train the model starting at this game state
            if self.allow_retraining: # and all([strat[s] == strat[s+1] for s in range(len(strat)-1)]):
                print("") #so we can see epoch training happen below the game progress

                #get original infoset
                info_set = self.regret_man.get_information_set(game_state, self, False)
                print("Current Strategy: {} - {}r | {}w".format(info_set.get_average_strategy(),info_set.reads(),info_set.writes()))
                path = self.regret_man.game_abstractor.gen_regret_path(game_state,self)
                print("Path: " + str(path))

                #request shared training (if shared training is not running, we will just run a single)
                self.focus_training(game_state,REGRETBOT_FOCUSTIME)

                #get updated infoset
                strat = info_set.get_average_strategy()
                print("Retrained Strategy: {} - {}r | {}w".format(strat,info_set.reads(),info_set.writes()))
        else:
            strat = np.zeros(len(valid_actions))

        #save a copy of strategy for verbose mode
        org_strat = strat.copy()

        #to calculate EV we need to know: pot size and total invested amount
        #pot_size = round_state["pot"]["main"]["amount"]
        invested_amt = 0
        for round in round_state["action_histories"]:
            for action in round_state["action_histories"][round]:
                if action["uuid"] == "p0":
                    #try to get paid amount (this will be available for raises, but not BB or SB)
                    #if its not there use amount (but only for BB / SB)
                    invested_amt += action.get("paid",action.get("amount",0))
        #invested_amt = sum([action["amount"] for action in [round_state["action_histories"][ah] for ah in round_state["action_histories"]][0] if action['uuid'] == 'p0'])

        #should we be using monte?
        if self.use_monte:
        
            #get a pre-calculated monte carlo simulation for this hole/comm combination
            (wins,games,win_rate) = self.abstractor.get_monte(hole_card,community_card)

            #get raw amounts of each action
            amounts = np.array([action['amount'] for action in valid_actions])

            #the action amounts for big blind and small blind are incorrect
            if round_state["street"] == "preflop":
                if round_state["big_blind_pos"] == 0:            
                    amounts[1] -= round_state["small_blind_amount"] * 2
                elif round_state["big_blind_pos"] == 1:
                    amounts[1] -= round_state["small_blind_amount"]

            #calculate upsides and downsides (first pass)
            upsides = (amounts * 2 + pot) * win_rate

            #adjust for call - it doesn't get amount*2, just amount
            upsides[1] = (amounts[1] + pot) * win_rate

            #calculate downsides
            downsides = (amounts + invested_amt) * (1 - win_rate)


            #we need to adjust invalid amounts and FOLD amounts
            for c in range(len(amounts)):
                #fold should be ZERO upside and INVESTED_AMT downside
                if c == 0:
                    upsides[c] = 0
                    downsides[c] = invested_amt

                #invalid amounts should be zero up and zero down
                if amounts[c] < 0:
                    upsides[c] = 0
                    downsides[c] = 0

            #normalize the upsides
            if sum(upsides) != 0: net_upsides = upsides / sum(upsides)
            else: net_upsides = upsides

            #normalize the downsides
            if sum(downsides) != 0: net_downsides = downsides / sum(downsides)
            else: net_downsides = downsides

            #adjust nets
            nets = upsides - downsides

            #adding minimum ensures all nets are at least at zero
            #so we can normalize them
            #but first we need to clear out the invalid nets
            if min(nets) < 0:
                #this should never happen for folding
                #because if folding is net ZERO it is because we can fold for free
                #not because folding is invalid (it's never INVALID)
                for n in range(1,len(nets)):
                    if nets[n] == 0:
                        nets[n] = min(nets)

            #now shift everything up so the lowest net is now at ZERO (if its negative)
            #then normalize
            nets += abs(min(nets))
            nets /= sum(nets)

            #get the community monte (this is the likelyhood of someone calling our raise, etc)

        #if we are not using CFR, clear the strategy
        #reset as normalized (untrained) strategy
        if not self.use_cfr: strat[:] = 1/len(strat)

        #if we are using CFR but the strategy is untrained
        #and we are saving untrained states for later review, save it now
        if self.use_cfr and max(strat) - min(strat) <= 0.01 and self.save_untrained_states:
            print("Saving Untrained State...")
            with open("/untrained/{}.gamestate".format(int(time.time())),"wb") as f:
                pickle.dump(game_state,f)

        #if we are using monte, apply as a layer over strategy
        if self.use_monte:

            #now, we can apply those upsides and downsides as a filter against our strategy
            strat += nets

            #finally - filter out negative strategies and then normalize them
            strat = np.maximum(0,strat)
            #strat /= sum(strat)

        #now, for all strategies, shift down as needed
        #only replace the strategy if our replacement is bigger
        #do not sum them - that could throw off our statistics
        for c in range(len(strat)-1,3,-1):
            if not valid_actions[c].get("valid",True):
                if strat[c] > strat[c-1]: strat[c-1] = strat[c]
                strat[c] = 0

        #if min raise is not valid disable it
        if not valid_actions[3].get("valid",True): strat[3] = 0

        #renormalize remaining strategies
        strat /= sum(strat)
                
        #pick best action
        #get the action from strategy
        if self.use_argmax:
            #use argmax to pick an action from our strategy
            best_action_index = np.argmax(strat)
        else:

            #get the top N strategies and their corresponding actions
            topindexes = np.argpartition(strat,REGRETBOT_WEIGHTEDSTRATS * -1)[REGRETBOT_WEIGHTEDSTRATS * -1:]
            topstrats = np.take(strat,topindexes)
            
            #pick a random action from those top strategies
            #but weight based on the strategy
            best_action_index = random.choices(range(len(topstrats)), weights=topstrats, k=1)[0]

            #that index is actually the index within topstrats, so use topindexes to get the original index
            best_action_index = topindexes[best_action_index]

        #do not allow folding when calling is free
        #JBC: 12/9/2020 - moved this from manipulating strategy because this would
        #sometimes cause a raise (with a negative EV) to be selected over folding
        if valid_actions[1]["amount"] == 0 and best_action_index == 0: 
            best_action_index = 1

        #best_action = self.choose_best_action(streets.index(street),hole_card,community_card,valid_actions) 
        #best_action_index = action_names.index(best_action)
        best_action = action_names[best_action_index]
        
        #temp fix - the average raise may not be valid
        action_amount = valid_actions[best_action_index]["amount"]
        
        #verbose - print out our bet
        if self.verbose:
            print("=====================================================")
            print("INVESTED: " + str(invested_amt) + ", POT: " + str(pot))
            print("STRAT: " + str(org_strat))
            print("MONTE: {} of {} = {:4.2f}%".format(wins,games,win_rate*100))
            print("AMTS: " + str(amounts))
            print("UPSIDES: " + str(upsides))
            print("DOWNSIDES: " + str(downsides))
            print("NETS: " + str(upsides - downsides))
            print("NETS %: " + str(nets))
            print("STRAT: " + str(strat))
            print("ACTION: " + str(best_action) + " " + str(action_amount))
            print("=====================================================")

        #return that action
        return best_action, action_amount

    def _setup_game_state(self, round_state, my_hole_card):
        game_state = restore_game_state(round_state)
        player_uuids = [player_info['uuid'] for player_info in round_state['seats']]
        game_state['table'].deck.shuffle()
        for uuid in player_uuids:
            if uuid == self.uuid:
                game_state = attach_hole_card(game_state, uuid, gen_cards(my_hole_card))  # attach my holecard
            else:
                game_state = attach_hole_card_from_deck(game_state, uuid)  # attach opponents holecard at random
        return game_state

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        #if we are on first street, update street start stack
        #which we use to calculate regret
        if round_state["street"] == "preflop":
            self.street_start_stack = [player for player in round_state['seats'] if player["uuid"] == self.uuid][0]["stack"]


    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        #count rounds played
        self.rounds += 1

def setup_ai():

    #fix printing issues
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    cfr_bot = EmulatorPlayer()
    cfr_bot.verbose = True
    cfr_bot.regret_man.load()
    cfr_bot.allow_retraining = True #we can retrain on the fly while we are playing
    return cfr_bot
