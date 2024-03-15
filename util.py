import random
import numpy as np
from sys import stdout
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
from pypokerengine.engine.hand_evaluator import HandEvaluator as he
from pypokerengine.engine.card import Card
from pypokerengine.engine.deck import Deck

#this function sets up emulator player structure from an array and returns it
def setup_emulator_players(emulator, players, uuids = [], stacks = [1000] ):
    #fix for stacks not matching player length
    if len(stacks) == 1: stacks = stacks * len(players)
    if len(uuids) == 0: uuids = ["p{}".format(c) for c in range(len(players))]

    #setup the players
    player_info = {}
    for c in range(len(players)):

        #setup uuid
        player = players[c]
        player_uuid = uuids[c]
        player.uuid = player_uuid

        #add the player to our emulator and various tracking structures
        player_info[player.uuid] = {"name":player.uuid, "stack": stacks[c], "algorithm": player}
        emulator.register_player(player.uuid, player)

    #return that player info
    return player_info

#generic progress bar
def progressBar(prefix, current, total, suffix = "",barLength = 50):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    output = prefix + ': [%s%s] %d %%: %s' % (arrow, spaces, percent, suffix)
    output += ' ' * (110-len(output))
    print(output, end='\r')
    stdout.flush()


#Estimate the ratio of getting a specific hand given a starting set of cards
def estimate_prob(nb_simulation, hole_card, community_card=None):
    if not community_card: community_card = []

    # Make lists of Card objects out of the list of cards
    initial_community_card = gen_cards(community_card)
    hole_card = gen_cards(hole_card)
    hit_count = 0
    hand_hits = {
        "HIGHCARD":0,
        "ONEPAIR": 0,
        "TWOPAIR": 0,
        "THREECARD":0,
        "STRAIGHT":0,
        "FLASH":0,
        "FULLHOUSE":0,
        "FOURCARD":0,
        "STRAIGHTFLASH":0
    }
    
    #for the number of simulations
    for i in range(nb_simulation):
        #fill community cards
        community_card = _fill_community_card(initial_community_card, used_card=hole_card + initial_community_card)
        
        #get hand rank info
        hand_rank = he.gen_hand_rank_info(hole_card, community_card)
        hand_hits[hand_rank["hand"]["strength"]] += 1

    # Estimate the win count by doing a Monte Carlo simulation
    for hand in hand_hits.items():
        print("{}: {:2.2f}".format(hand[0] , 100 * hand[1] / nb_simulation))

#the shape of the NP array used for multi monte array
def multi_monte_shape():
    return [4,1601,1600,2]

def multi_monte(abstractor,players, simulations, results = []):
        
    #setup our results dictionary if not passed to us
    if len(results) == 0: results = np.zeros(multi_monte_shape(),'i')
    
    #get a deck of cards
    deck = Deck()
    
    #for each simulation
    print("Building Monte Carlo Simulations")
    for sim in range(simulations):
        
        #print progress every 2%
        if sim % (simulations / 50) == 0: progressBar("Progress",sim,simulations)

        #shuffle the deck
        deck.shuffle()
        
        #get holes and strengths for final game state
        holes, strengths, wins, hands, profiles = [], [], [], [], []
        winner = 0
        
        #community doesn't change for each player
        community = deck.deck[players * 2-1:players * 2 - 1 + 5]
        
        #calculate holes, strengths and winner in ONE loop
        for p in range(players):
            #figure out this hole
            hole = deck.deck[2*p:2*p+2]
            strength = he.eval_hand(hole, community)
            if strength > winner: winner = strength
                
            #calculate hole rank
            hole_rank = he.gen_hand_rank_info(hole,[])
            hole_hand = hole_rank["hand"]["strength"]
            hole_bucket = abstractor.get_hand_strength_bucket(hole_hand,hole,[])
                
            #append to both arrays, convert cards to strengths
            hands.append(hole_hand)
            profiles.append(abstractor.hand_profile(hole,[],hole_rank))
            holes.append(hole)
            strengths.append(strength)
            wins.append(0)

        #now set winners (sadly in another loop)
        for p in range(players):
            if strengths[p] == winner: wins[p] = 1
                
        #for each street, generate hand profiles
        for street in [0,3,4,5]:
            
            #calc street index,
            street_index = street - 2
            if street_index < 0: street_index = 0
            
            #for each player
            #generate their hand profile at this street
            for p in range(players):
                
                #get hand ranking for hole plus street cards
                rank = he.gen_hand_rank_info(holes[p],community[:street])
                profile = abstractor.hand_profile(holes[p] , community[:street],rank)

                #calculate the community card profile
                comm_profile = 0
                if street > 0:
                    comm_rank = he.gen_hand_rank_info(community[:2],community[2:street])
                    comm_profile = abstractor.hand_profile(community[:2],community[2:street],comm_rank)
                
                #testing
                #print("h {}{} | c {}{} = {} vs {}".format(profile,[str(c) for c in holes[p]],comm_profile, [str(c) for c in community[:street]],results[street_index][profile][-1][1],results[street_index][profile][comm_profile][1]))
                
                #update the hole overall win rate - # of wins and # of participations
                results[street_index][profile][-1][0] += 1
                results[street_index][profile][-1][1] += wins[p]

                #update the community card win rate - # of wins and # of participations
                #under this hole
                results[street_index][profile][comm_profile][0] += 1
                results[street_index][profile][comm_profile][1] += wins[p]
    return results
    
    