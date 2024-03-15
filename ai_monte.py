from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
from PokerAbstractor import PokerAbstractor

# Estimate the ratio of winning games given the current state of the game
def estimate_win_rate(nb_simulation, nb_player, hole_card, community_card=None):
    if not community_card: community_card = []

    # Make lists of Card objects out of the list of cards
    community_card = gen_cards(community_card)
    hole_card = gen_cards(hole_card)

    # Estimate the win count by doing a Monte Carlo simulation
    win_count = sum([montecarlo_simulation(nb_player, hole_card, community_card) for _ in range(nb_simulation)])
    return 1.0 * win_count / nb_simulation


def montecarlo_simulation(nb_player, hole_card, community_card):
    # Do a Monte Carlo simulation given the current state of the game by evaluating the hands
    community_card = _fill_community_card(community_card, used_card=hole_card + community_card)
    unused_cards = _pick_unused_card((nb_player - 1) * 2, hole_card + community_card)
    opponents_hole = [unused_cards[2 * i:2 * i + 2] for i in range(nb_player - 1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0


class DataBloggerBot(BasePokerPlayer):
    def __init__(self, abstractor = None):
        super().__init__()
        self.wins = 0
        self.losses = 0
        self.abstractor = abstractor #if no abstractor is defined we will need to calculate monte the old fashioned way

    def declare_action(self, valid_actions, hole_card, round_state):

        #if an abstractor is not defined, estimate win rate
        #otherwise - use our very precise precalculated win rate (run over hundreds of millions of simulations)
        if self.abstractor == None:
            win_rate = estimate_win_rate(50, self.num_players, hole_card, round_state['community_card'])
        else:        
            (wins,games,win_rate) = self.abstractor.get_monte(hole_card,round_state['community_card'])


        # figure out the pot size
        pot_size = round_state["pot"]["main"]["amount"]
        street = round_state["street"]

        #let someone know what we are doing
        #print("---MONTE ACTION---")
        #print("street = " + street)
        #print("hole card = " + str(hole_card))
        #print("pot size = " + str(pot_size))
        #print("win ratio = " + str(win_rate))

        # Check whether it is possible to call
        can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
        if can_call:
            # If so, compute the amount that needs to be called
            call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
        else:
            call_amount = 0

        amount = None
        
        #calculate pot odds
        call_pot_odds = call_amount / ( call_amount + pot_size )
        
        #don't cause division by zero
        if win_rate == 0.9: win_rate = 0.91

        #calculate max profitable raise and raise pot odds
        max_profitable_raise = ( pot_size * ( win_rate + 0.1) ) / ( 0.9 - win_rate )
        
        #what is the call amount?
        #print("call amount = " + str(call_amount))
        #print("call pot odds = " + str(call_pot_odds))
        #print("max raise = " + str(max_profitable_raise))        

        #JBC figure out min and max raise options
        raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
        max_raise = raise_amount_options['max']
        min_raise = raise_amount_options['min']

        #is max raising profitable
        if win_rate > max_raise / (pot_size + max_raise):
            action = 'raise'
            amount = max_raise
        elif win_rate > min_raise / (pot_size + min_raise):
            action = 'raise'
            amount = min_raise
        elif win_rate > call_pot_odds:
            action = 'call'
        else:
            action = 'call' if can_call and call_amount == 0 else 'fold'

        # Set the amount
        if amount is None:
            items = [item for item in valid_actions if item['action'] == action]
            amount = items[0]['amount']

        return action, amount

    def receive_game_start_message(self, game_info):
        self.num_players = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        is_winner = self.uuid in [item['uuid'] for item in winners]
        self.wins += int(is_winner)
        self.losses += int(not is_winner)


def setup_ai():
    return DataBloggerBot()
