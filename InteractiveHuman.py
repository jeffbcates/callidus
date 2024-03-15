from pypokerengine.players import BasePokerPlayer

class InteractiveHuman(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #static class variables
    QUIT_ALL = False

    #init player
    def __init__(self, verbose = True):
        self.verbose = verbose

    #get our name from round state
    def get_name(self,round_state, uuid):

        #find us in seats
        for s in round_state["seats"]:
            if s['uuid'] == uuid: return s['name']
    
    #we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):

        #if we are in a "quitting" state
        #just return fold
        if InteractiveHuman.QUIT_ALL:
            return "fold",0

        #get our name
        name = self.get_name(round_state, self.uuid)

        #get actions
        for act in valid_actions:
            if act["action"] == "call": callamt = act["amount"]
            if act["action"] == "raise": 
                raisemin = act["amount"]["min"]
                raisemax = act["amount"]["max"]
        
        #display valid actions
        print("{} - quit, fold, call {}, or raise {} - {}?: ".format(name,callamt,raisemin,raisemax), end="")

        #prompt user for action
        user_input = input().split()
        action = user_input[0].lower()

        #return our action
        amount = 0
        if action == "f":
            action = "fold"
        if action in ("q","quit"):
            InteractiveHuman.QUIT_ALL = True
            action = "fold"
        if action in ("c","call"): 
            amount = float(callamt)
            action = "call"
        if action in ("r","raise"): 
            amount = float(user_input[1])
            action = "raise"
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
