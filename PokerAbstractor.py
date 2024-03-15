#generic libraries
import numpy as np
import random
import pickle

#implement our game abstractor
from GameAbstractor import GameAbstractor
from Regrets import RegretManager

#for PyPokerEngine specific functionality
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.hand_evaluator import HandEvaluator as he
from pypokerengine.utils.card_utils import gen_cards, gen_deck
from pypokerengine.engine.card import Card
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck

#needed for opening shared monte
from multiprocessing import shared_memory
from util import multi_monte_shape

class PokerAbstractor(GameAbstractor):

	ABSTRACTED_BB = 10 #big blind
	ABSTRACTED_RAISES = 4 #different raise options, PLUS all-in (ZERO)

	#when true all bets for the table are summarized into a single action
	SIMPLE_BETS = True

	#the following are how we abstract bets from other players when tracking them
	ABSTRACTED_BETS = [(2,0),(4,1),(8,2),(16,3),(32,4),(64,5),(128,6),(256,7),(512,8),(1024,9)]

	#abbreviated hand strengths
	ABB_STRENGTH = { "HIGHCARD":"HC", "ONEPAIR":"1P", "TWOPAIR":"2P", "THREECARD":"3C", "STRAIGHT":"ST","FLASH":"FL","FULLHOUSE":"FH","FOURCARD":"4C","STRAIGHTFLASH":"SF", "NONE":"NA"}

	#hands (as misspelled by pypokerengine) along with their buckets (i.e. indexes)
	hands = ["HIGHCARD","ONEPAIR","TWOPAIR","THREECARD","STRAIGHT","FLASH","FULLHOUSE","FOURCARD","STRAIGHTFLASH"]
	hand_buckets = {"HIGHCARD":0,"ONEPAIR":1,"TWOPAIR":2,"THREECARD":3,"STRAIGHT":4,"FLASH":5,"FULLHOUSE":6,"FOURCARD":7,"STRAIGHTFLASH":8}

	#play histories
	playhistory_names = ["O","S","B","C","R0","R1","R2","R3","R4","R5","R6","R7","R8","R9"]
	playhistory_buckets = {}

	#as a quick reference, how many community cards are present for each street
	street_community_sizes = [0,3,4,5,5,0] #index is street (0=preflop ,etc)

	#abstraction of player positions
	#note: this assumes the player is always ordinal zero
	#we assume the ordinal of inner list is the position of dealer and value is our abstracted position
	pos_buckets = [
		[2,0], #2 players (dealer is player, so LP, other player is EP)
		[2,1,0], #3 players
		[2,0,2,1], #4 player 
		[2,1,0,2,2], #5 players
		[2,2,1,0,2,2], #6 players
		[2,1,1,0,0,2,2], #7 players
		[2,1,1,0,0,0,2,2], #8 players
		[2,1,1,1,0,0,0,2,2] #9 players
	]

	#initialize the abstractor
	def __init__(self):

		#pass to super - save game reference
		super().__init__()

		#build and save strength buckets for later use
		self.strength_buckets = self.build_strength_buckets()

		#build play history buckets from names
		for c in range(len(self.playhistory_names)):
			#get the name of this action
			villian_action = self.playhistory_names[c]
			hero_action = "P-" + villian_action

			#villian action exists at leaf path 1
			self.playhistory_buckets[villian_action] = [(1,RegretManager.LEAF_PATH),(c,RegretManager.ACTION_PATH)]

			#hero action exists at leaf path 2
			self.playhistory_buckets[hero_action] = [(2,RegretManager.LEAF_PATH),(c,RegretManager.ACTION_PATH)]			

		#monte carlo simulator starts empty until its loaded
		self.monte_results = []

	#build strength buckets
	def build_strength_buckets(self):
        #get our low and high strengths for each hand type
		strength_buckets = {
            "HIGHCARD" : [
                he.eval_hand(gen_cards(["SA","DK"]),[]),
                he.eval_hand(gen_cards(["S2","D3"]),[])        
            ],
            "ONEPAIR" : [
                he.eval_hand(gen_cards(["SA","DA"]),[]),
                he.eval_hand(gen_cards(["S2","D2"]),[])        
            ],
            "TWOPAIR" : [
                he.eval_hand(gen_cards(["SA","DA","SK","DK"]),[]),
                he.eval_hand(gen_cards(["S2","D2","S3","D3"]),[])
            ],
            "THREECARD" : [
                he.eval_hand(gen_cards(["SA","DA","CA"]),[]),
                he.eval_hand(gen_cards(["S2","D2","C2"]),[])

            ],
            "STRAIGHT" : [
                he.eval_hand(gen_cards(["SA","DK","CQ","HJ","ST"]),[]),
                he.eval_hand(gen_cards(["SA","D2","C3","H4","S5"]),[])

            ],
            "FLASH" : [
                he.eval_hand(gen_cards(["SA","SK","SQ","SJ","S9"]),[]),
                he.eval_hand(gen_cards(["S2","S3","S4","S5","S7"]),[])

            ],
            "FULLHOUSE" : [
                he.eval_hand(gen_cards(["SA","DA","CA","SK","DK"]),[]),
                he.eval_hand(gen_cards(["S2","D2","C2","S3","D3"]),[])

            ],
            "FOURCARD" : [
                he.eval_hand(gen_cards(["SA","DA","CA","HA"]),[]),
                he.eval_hand(gen_cards(["S2","D2","C2","H2"]),[])

            ],
            "STRAIGHTFLASH" : [
                he.eval_hand(gen_cards(["SA","SK","SQ","SJ","ST"]),[]),
                he.eval_hand(gen_cards(["SA","S2","S3","S4","S5"]),[])

            ]
        }

        #step through each hand type and add the X buckets     
		return strength_buckets

	#gen starting holes - returns every combination of hole
	#that satisfies the specified stregnth
	def gen_starting_holes(self, strength, bucket):

		#get a new deck
		deck_manager = gen_deck()
		deck = deck_manager.deck
		holes = []

		#for each card in the deck
		for c1 in range(len(deck)):
			for c2 in range(c1+1,len(deck)):
				#get rank of this cardset
				hole = [deck[c1],deck[c2]]
				hole_rank = he.gen_hand_rank_info(hole,[])
				hole_strength = hole_rank["hand"]["strength"]
				hole_bucket = self.get_hand_strength_bucket(hole_strength,hole,[])
				
				#if this is what we are looking for, add to return
				if hole_strength == strength and hole_bucket == bucket: 
					holes.append(hole)

		#return those holes
		return holes

    #get hand bucket for hand / strength
	def get_hand_strength_bucket(self, strength, hole_card, community_card):
		#build hand strength bucket given cards
		high,low = self.strength_buckets[strength]
		hand_eval = he.eval_hand(hole_card,community_card)
		bucket_size = int((high-low)/9)
		hand_bucket = int( (hand_eval - low) / bucket_size)
		return hand_bucket

	#given an abstracted position and # of players, what would the button
	#position be that would make player ZERO match that position
	def unpack_button_pos(self,pos,players):

		#with the array of possible positions based on # of players
		positions = self.pos_buckets[players-2]
		for cx in range(len(positions)):
			if positions[cx] == pos:
				return cx

	#unpack an action name/amount given the packed action along with current game state
	def unpack_action(self, entry, valid_actions):

		#get action name from play history 
		action_name = self.playhistory_names[entry[0]]

		#small and big blind actions -> just return NONE
		if action_name in ['S','B']: return (None,None)

		#out action -> return fold
		if action_name == 'O': return ("fold",0)

		#call action -> return the current call amount
		if action_name == "C": return ("call",valid_actions[1]["amount"])

		#for raise, unpack the abstracted raise amount and then return that as a raise
		amount = self.unpack_abstracted_amount(entry[0] - 4)
		
		#get min / max raise amounts
		min_raise = valid_actions[2]["amount"]["min"]
		max_raise = valid_actions[2]["amount"]["max"]

		#check for disallowed action
		assert min_raise > 0 and max_raise > 0, "Disallowed Raising Not Currently Implemented"
		
		#adjust to be within raise bounds
		amount = min(max(amount,min_raise),max_raise)

		#return amount and action
		return ("raise",amount)

	#unpack a hand profile to a tuple
	def unpack_hand_profile(self,profile):

		#the profile "-1" means an empty set of cards
		if profile == -1: return ("NONE",0,0,0)

		#unpack the profile

		hand_bucket = profile % 10
		strength_bucket = (profile - hand_bucket) % 100 // 10
		remainder = (profile - hand_bucket - strength_bucket) // 100
		fcount = remainder % 4 + 2
		scount = remainder // 4 + 2

		if hand_bucket >= len(self.hands):
			print("unpack_hand_profile({}): hand bucket {} out of range using zero".format(profile,hand_bucket))
			hand_bucket = 0

		return (self.hands[hand_bucket],strength_bucket,fcount,scount)

	#generate hand probabilities for the hand used in abstraction
	#this is a lookup on pre-computed probabilities for different ahnd types
	#and have been normalized and rangemaxed to integer 0..9
	#packed_profile -> if False returns a tuple with the values that makeup the profile, true returns just the index
	def hand_profile(self,hole, community,hand_rank = None, packed_profile = True):

		#we allow hand profiles for the community cards when no hole is provided
		#this adjusts for that here
		if len(hole) == 0:
			if len(community) > 0: 
				hole = community[:2]
				community = community[2:]
			else: return -1

		#if no hand rank is defined, lookup here
		if hand_rank == None: hand_rank = he.gen_hand_rank_info(hole,community)

		#with one pass, count all suits and figure out gaps for straights
		#Heart=8, Spade=16, Diamond=4, Club=2
		hearts, spades, diamonds, clubs = 0,0,0,0

		#combine all cards for our calcs
		cards = hole + community

		#check all cards for flush, and build the straight ranks in ONE loop
		ranks = []
		high = 0
		low = 15
		slow, shigh = 0, 0
		for c in cards:
			#count all suits up to the first 5 of each (no need afer that)
			clubs += c.suit == 2 and clubs < 5
			diamonds += c.suit == 4 and diamonds < 5
			hearts += c.suit == 8 and hearts < 5
			spades += c.suit == 16 and spades < 5

			#for straight - do some calculations
			#ace can also be low so we add that as "1" as well
			ranks.append(c.rank)        
			if c.rank == 14:
				low = 1
				ranks.append(1)
			if c.rank < low: low = c.rank
			if c.rank > high: high = c.rank

		#calculate flush probs
		fcount = max([clubs,diamonds,hearts,spades])

		#strip out just card ranks to calculate straight count
		ranks = set(ranks)
		for r in ranks:
			#if within 5 from low
			slow += r <= low+5 and r >= low
			shigh += r >= high-5 and r <= high

		#calculate straight probs
		#for sneeds, don't show a negative 
		scount = max([slow,shigh])
		sneeds = max(0,5 - scount)

		#don't let either number go above 5
		if fcount > 5: fcount = 5
		if scount > 5: scount = 5

		#get hand strength bucket and hand bucket
		hand = hand_rank["hand"]["strength"]
		hand_bucket = self.hand_buckets[hand]
		strength_bucket = self.get_hand_strength_bucket(hand,hole,community)

		#to cut the number of buckets in almost 1/2 we combine 1 and 2 card flush/straight draws
		fcount -= 2
		if fcount < 0: fcount = 0
		scount -= 2
		if scount < 0: scount = 0

		#build profile from those values    
		profile = hand_bucket + strength_bucket * 10 + (fcount + scount * 4) * 100

		#return flush count, straight count, and straight needs
		if packed_profile:
			return hand_bucket + strength_bucket * 10 + (fcount + scount * 4) * 100
		else:
			return (hand_bucket,strength_bucket,(fcount + scount * 4))

	#given game state, figure out the current pot
	#since this is not immediately obvious from the game state
	def calculate_current_pot(self, table):
		#add up the paid amounts for all players in the game (active or not)
		return sum([p.paid_sum() for p in table.seats.players])

	#normalize an amount by BB
	def abstract_amount(self, amount, bigblind = None):
		#use default if needed
		if bigblind == None or bigblind == 0: bigblind = PokerAbstractor.ABSTRACTED_BB

		#translate into # of big blinds
		amount = int(amount / bigblind)

		#now abstract bet amount further -> each tuple contains :: ( max bet in range, abstracted amount)
		#if the bet is greater than 1024, then abstracted amount 9 is used
		bet = min([b for b in PokerAbstractor.ABSTRACTED_BETS if b[0] > amount] + [PokerAbstractor.ABSTRACTED_BETS[-1]])
		abstracted = bet[1]
		return abstracted

	#unpack an abstracted bet amount to the bet exactly at the minimum of that range
	def unpack_abstracted_amount(self, abstracted_amount, bigblind = None):
		#use default if needed
		if bigblind == None or bigblind == 0: bigblind = PokerAbstractor.ABSTRACTED_BB

		#get our two amounts
		max_amt = PokerAbstractor.ABSTRACTED_BETS[abstracted_amount][0] * bigblind
		if abstracted_amount > 0: min_amt = PokerAbstractor.ABSTRACTED_BETS[abstracted_amount-1][0] * bigblind
		else: min_amt = 0

		#translate into # of big blinds
		amount = min_amt + ( max_amt - min_amt ) // 2

		#return that amount
		return amount

	#get hand probability from our pre-calculated monte carlo model
	#we always return an integer value between 0 and 100 (the rounded % of probability)
	def get_hand_prob(self, street, hole_profile, community_profile):

		#lookup the probability profile
		games = self.monte_results[street][hole_profile][community_profile][0]
		wins = self.monte_results[street][hole_profile][community_profile][1]

		#if there are no games, return nothing
		if games == 0: return 0

		#return the win ratio for this hole/community combination
		return wins * 100 // games

	#flatten a play history array for a single player and round
	#they may have raised/called multiple times in a round, we just look at total
	def flatten_action_history(self,action_history, player_pos):

		#total bet amount and final action
		flat_history = ""

		#step through action history
		for a in action_history:
			#add the action amount
			#note that some actions (fold) have no amount
			bet_amount = a.get("amount",0)
			final_action = a.get("prefix","") + a["action"] # we add a prefix to self-play actions so we can find them later

			#clean up action name -> treat "FOLD" and "OUT" the same, as after the round "FOLD" shows as "OUT" instead
			final_action = final_action.replace("SMALLBLIND","S").replace("BIGBLIND","B").replace("FOLD","O").replace("CALL","C").replace("RAISE","R").replace("OUT","O")

			#add bet amount to final action and return (abstract bet amount into a range of 0..9)
			if final_action == "R": final_action += str(self.abstract_amount(bet_amount))

			#add player flag if this is the players action
			if a["pos"] == player_pos: final_action = "P-" + final_action

			#add to flat history
			flat_history += final_action

		#return the flatten history
		return flat_history

	#flatten a single action history
	def flatten_action(self,action,player_pos):

		#add the action amount
		#note that some actions (fold) have no amount
		bet_amount = action.get("amount",0)
		final_action = action.get("prefix","") + action["action"] # we add a prefix to self-play actions so we can find them later

		#clean up action name -> treat "FOLD" and "OUT" the same, as after the round "FOLD" shows as "OUT" instead
		final_action = final_action.replace("SMALLBLIND","S").replace("BIGBLIND","B").replace("FOLD","O").replace("CALL","C").replace("RAISE","R").replace("OUT","O")

		#add bet amount to final action and return (abstract bet amount into a range of 0..9)
		if final_action == "R": final_action += str(self.abstract_amount(bet_amount))

		#add player flag if this is the players action
		if action["pos"] == player_pos: final_action = "P-" + final_action

		#return the flatten history
		return final_action

	#sort play history by occurance (instead of by player)
	#around a player position - ie if we are player 1 and button is player 2 > [ [p3,p4,p5,p0,p1] ... ]
	def sort_action_histories(self, action_histories,player_pos,btn_pos):

		#figure out the lengths of the longest action histories
		longest_history = max([len(h) for h in action_histories])
		players = len(action_histories)

		#step through all round action histories and sort them in order
		action_history = [[]]
		c = 0
		h = 0
		while c < longest_history:

			#step through all players and add this action history (if available)
			#action_history.append([])
			p = btn_pos+1
			players_checked = 0
			while players_checked < players:
				#cycle player back to start if at end of table      
				if p >= len(action_histories): p = 0

				#setup a new history, include position for debuggin
				if c < len(action_histories[p]):
					new_hist = action_histories[p][c]
					new_hist['pos'] = p
				else:
					new_hist = None

				#add player history, but if not there use "OUT" as history
				#and do not add our own history here
				if p != player_pos and new_hist != None:
					action_history[h].append(new_hist)

				#if we have encountered our player, start a new action set
				if p == player_pos: 
					#if the player action is not "OUT"
					if new_hist != None:
						h += 1
						action_history.append([new_hist])

					#move to the next section (we don't group our actions along with other players)
					h += 1
					action_history.append([])

				#move to next history
				#and cycle player back to start if at end of table      
				players_checked += 1
				p += 1
				if p >= len(action_histories): p = 0

			#move to next set of actions
			c += 1

		#we have added all round history
		return action_history

	#simplify action histories of players so that only the highest raise is returned for each action set
	def simplify_action_histories(self, action_history):

		#step through each action history
		for h in range(len(action_history)):

			#if this action history has items
			if len(action_history[h]) > 0:

				#assume the first action history is the best
				best_hist = action_history[h][0]

				#step through all actions in this history
				for a in action_history[h]:

					#is the amount greater than current best?
					if a.get("amount",0) > best_hist.get("amount",0): best_hist = a
					elif best_hist["action"] == "FOLD" or best_hist["action"] == "OUT": best_hist = a

				#update action history to include only the one history
				action_history[h] = [best_hist]

		#return the simplified action histories
		return action_history

	#abstract player history
	#JBC - need to optimize this (we are making many passes through play history here)
	#this modified version combines sort_action_histories, simplify_action_histories, and converting to tuples in one
	def abstract_play_history(self, street, table, player_seat, player_pos):

		#save off dealer button position
		btn_pos = table.dealer_btn

		#are we dealing with round histories or action histories
		pull_round_histories = not ( table.seats.players[0].round_action_histories[street] == None )

		#build all our play histories
		#and while doing so, figure out the longest history
		#so we don't have to loop back through a second time for that
		play_histories = []
		longest_history = 0 
		for p in table.seats.players:

			#add either round histories or action histories
			if pull_round_histories:
				play_histories.append(p.round_action_histories[street])
			else:
				play_histories.append(p.action_histories)

			#is this our longest history?  if so, update it
			if len(play_histories[-1]) > longest_history: longest_history = len(play_histories[-1])

		###THE FOLLOWING SORTS THE PLAY HISTORIES WE JUST BUILT###

		#step through all round action histories and sort them in order
		players = len(play_histories)
		action_history = [[]]
		simplified_action_history = [[]]
		flatten_history = []
		play_history_tuples = []
		villian_last = False
		c = 0
		h = 0
		while c < longest_history:

			#step through all players and add this action history (if available)
			p = btn_pos+1
			players_checked = 0
			while players_checked < players:
				#cycle player back to start if at end of table      
				if p >= len(play_histories): p = 0

				#setup a new history, include position for debuggin
				if c < len(play_histories[p]):
					new_hist = play_histories[p][c]
					new_hist['pos'] = p
				else:
					new_hist = None

				#add player history, but if not there use "OUT" as history
				#and do not add our own history here
				if p != player_pos and new_hist != None:
					#add to action history
					action_history[h].append(new_hist)
					
					#if there are no simplified histories for this round step
					#we should just add this one
					if len(simplified_action_history[h]) == 0:
						#no need to check, just add it
						simplified_action_history[h] = [new_hist]

						#flag that villian history was added after player
						#so that at the end of the main loop we can log its flattened action
						villian_last = True

					else:
						#get best history reference to make this code more readible
						best_hist = simplified_action_history[h][0]

						#there is already an action history, are we better than it?
						if new_hist.get("amount",0) > best_hist.get("amount",0): 
							simplified_action_history[h][0] = new_hist
						elif best_hist["action"] == "FOLD" or best_hist["action"] == "OUT": 
							simplified_action_history[h][0] = new_hist

				#if we have encountered our player, start a new action set
				if p == player_pos: 
					#if the player action is not "OUT"
					if new_hist != None:

						#flatten this history and add to flattened history array
						#also translate into our final tuple and add that to tuple array
						if len(simplified_action_history[h]) > 0:
							flatten_history.append(self.flatten_action(simplified_action_history[h][0], player_pos))
							play_history_tuples += self.playhistory_buckets[flatten_history[-1]]

						#move on to player history
						h += 1
						action_history.append([new_hist])

						#there is only ever one history for the player
						#so we can just add that now
						simplified_action_history.append([new_hist])

					#flatten this history and add to flattened history array
					if len(simplified_action_history[h]) > 0:
						flatten_history.append(self.flatten_action(simplified_action_history[h][0], player_pos))
						play_history_tuples += self.playhistory_buckets[flatten_history[-1]]

					#if we made it here then the villians last move has been recorded
					#and we do not need to record it at the end of the main loop
					villian_last = False

					#move to the next section (we don't group our actions along with other players)
					h += 1
					action_history.append([])
					simplified_action_history.append([])

				#move to next history
				#and cycle player back to start if at end of table      
				players_checked += 1
				p += 1
				if p >= len(play_histories): p = 0

			#move to next set of actions
			c += 1

		#final check - did we add the last villian action?
		if villian_last:
			flatten_history.append(self.flatten_action(simplified_action_history[h][0], player_pos))
			play_history_tuples += self.playhistory_buckets[flatten_history[-1]]


		#return the play history for this street
		return play_history_tuples

	#abstract player history
	#JBC - need to optimize this (we are making many passes through play history here)
	#this modified version combines sort_action_histories, simplify_action_histories, and converting to tuples in one
	def abstract_play_history_slow(self, street, table, player_seat, player_pos):

		#if the street is not in round histories then its the active street
		if table.seats.players[0].round_action_histories[street] == None:
			#use the current action histories - this is the active street
			play_histories = [p.action_histories for p in table.seats.players]
		else:
			#use round history for this street from each player
			play_histories = [p.round_action_histories[street] for p in table.seats.players]

		#sort play history around our player position
		sorted_play_history = self.sort_action_histories(play_histories,player_pos,table.dealer_btn)

		#if in simple mode, simplify the play histories
		if self.SIMPLE_BETS: sorted_play_history = self.simplify_action_histories(sorted_play_history)

		#collapse all play histories and turn them into tuples in a single pass
		play_history_tuples = []
		for h in sorted_play_history:
			if len(h) > 0:
				#get flattened play history
				flattened_history = self.flatten_action_history(h,player_pos)

				#add this action (this will include the action tuple with appropriate leaf tuple)
				play_history_tuples += self.playhistory_buckets[flattened_history]
		
		#return the play history for this street
		return play_history_tuples

	#covert a set of cards to an index value (most useful for hole cards)
	def pack_hole(self, cards):
		a = cards[0].to_id()
		b = cards[1].to_id()
		if a > b: c = a
		else: 
			c = b
			b = a
		return (b - 1) + (c - 1) * 52
		#card_ids = [cards[cx].to_id() for cx in range(len(cards))]
		#card_ids.sort()
		#return sum([(card_ids[cx]-1) * 52**cx for cx in range(len(card_ids))])

	#unpack cards from an integer value
	def unpack_hole(self, card_index):
		cards = [card_index % 52 + 1,card_index // 52 + 1]
		return [Card.from_id(cx) for cx in cards]

	#abstract the card set of the player
	#this can be combined with round info and opponent history info to create the regret path
	def abstract_hand_info(self, street, hole_card, community_card):

		#manipulate the community cards based on the street we are generating
		#even if its not the actual street based on # of cards (i.e. look back in time as needed)
		if street == 0: community_card = []
		if street == 1: community_card = community_card[:3]
		if street == 2: community_card = community_card[:4]
		if street == 3: community_card = community_card[:5]

		#return that abstracted hand
		if street == 0: 
			#get hole index
			hole_index = self.pack_hole(hole_card)

			#return only hole index
			return [(hole_index,RegretManager.HOLE_PATH)] #,(monte,2)
		else:

			#get the hand profile (packed_profile = False returns the tuple)
			hand_profile = self.hand_profile(hole_card,community_card, packed_profile = False)

			#return the hand profile
			return [
				(hand_profile[0], RegretManager.HAND_PATH),
				(hand_profile[1], RegretManager.STRENGTH_PATH),
				(hand_profile[2], RegretManager.SUITCOUNT_PATH)
			]

	#abstract player position information
	def abstract_position(self, table, player_pos):

		#determine our abstracted position
		pos = self.pos_buckets[table.seats.size()-2][table.dealer_btn]
		
		#return our abstracted position
		return [(pos, RegretManager.POS_PATH)]

    #generate a regret path given game state
	def gen_regret_path(self, game_state, player):
		#find our player seat at the table
		table = game_state["table"]
		player_seat = None
		player_pos = 0 #OPTIMIZE - lookup player position instead of looping
		for p in table.seats.players:
			if p.uuid == player.uuid:
				player_seat = p
				break
			else:
				player_pos += 1

		#get some generic information about the game state
		street = game_state["street"]
		total_players = table.seats.size()
		active_players = table.seats.count_active_players()
		
        #calculate strength and rank of hole
		hole_rank = he.gen_hand_rank_info(player_seat.hole_card,[])
		hole_strength = hole_rank["hand"]["strength"]
		hole_bucket = self.get_hand_strength_bucket(hole_strength,player_seat.hole_card,[])

		#get community card
		community_card = table.get_community_card()

		#all paths start with hole card info
		regret_path = self.abstract_hand_info(0, player_seat.hole_card, community_card)

		#add position abstraction
		regret_path += self.abstract_position(table,player_seat)
		
		#add play history
		regret_path += self.abstract_play_history(0,table,player_seat, player_pos)

		#add the different streets into path so we can replicate the path we took
		#notice that at each street we drop in our path tuple
		if street > 0:
			#before each street we need to add an end of street path
			#so that we can separate the play histories that happen to end earlier than those that take longer
			#for exmaple: HOLE > R1 P-C R3 P-C > FLOP -- vs -- HOLE > R1 P-C > FLOP
			#if we didn't have this we would mix up those two paths
			regret_path += [(3,RegretManager.LEAF_PATH)]

			#start the next street and add play history for it as well
			regret_path += self.abstract_hand_info(1, player_seat.hole_card, community_card)
			regret_path += self.abstract_play_history(1,table,player_seat, player_pos)
		if street > 1: 
			#before each street we need to add an end of street path
			#so that we can separate the play histories that happen to end earlier than those that take longer
			#for exmaple: HOLE > R1 P-C R3 P-C > FLOP -- vs -- HOLE > R1 P-C > FLOP
			#if we didn't have this we would mix up those two paths
			regret_path += [(3,RegretManager.LEAF_PATH)]

			#start next street
			regret_path += self.abstract_hand_info(2, player_seat.hole_card, community_card)
			regret_path += self.abstract_play_history(2,table,player_seat, player_pos)


		if street > 2: 
			#before each street we need to add an end of street path
			#so that we can separate the play histories that happen to end earlier than those that take longer
			#for exmaple: HOLE > R1 P-C R3 P-C > FLOP -- vs -- HOLE > R1 P-C > FLOP
			#if we didn't have this we would mix up those two paths
			regret_path += [(3,RegretManager.LEAF_PATH)]

			#start next street
			regret_path += self.abstract_hand_info(3, player_seat.hole_card, community_card)
			regret_path += self.abstract_play_history(3,table,player_seat, player_pos)

		#this is the only part of the path that's a leaf
		regret_path += [(0,RegretManager.LEAF_PATH)] #leaf path

        #return that path
		return regret_path

	#unpack a path
	def unpack_path_tuple(self,path):

		unpacked_path = []
		for p in path:
			if type(p) == tuple:
				if p[1] == RegretManager.HOLE_PATH:
					unpacked_path.append(" ".join([str(c) for c in self.unpack_hole(p[0])]))
				elif p[1] == RegretManager.POS_PATH:
					unpacked_path.append(["EP","MP","LP"][p[0]])
				elif p[1] == RegretManager.LEAF_PATH:
					if p[0] == 0: unpacked_path.append("LEAF")
					elif p[0] == 1: unpacked_path.append("VILLIAN")
					elif p[0] == 2: unpacked_path.append("HERO")
					else: unpacked_path.append("STREET")
				elif p[1] == RegretManager.ACTION_PATH:
					unpacked_path.append(self.playhistory_names[p[0]])
				elif p[1] == RegretManager.HAND_PATH:
					hand_profile = self.unpack_hand_profile(p[0])
					unpacked_path.append("".join([str(p) for p in hand_profile]))
				elif p[1] == RegretManager.STRENGTH_PATH:
					unpacked_path.append(str(p[0]))
				elif p[1] == RegretManager.SUITCOUNT_PATH:
					unpacked_path.append(str(p[0]))
			else:
				unpacked_path.append(p)
		return unpacked_path


	def gen_cards(self, cards):

		#just call our reference gen card function
		return gen_cards(cards)

	def abstract_game_state(self, game_state, player):
		#return our regret path
		return "".join([path + "|" for path in self.gen_regret_path(game_state,player)])[:-1]

	#extract an action name and value from a valid action for the game
	def abstract_action(self, action):
		#get action name
		return 1/0

	#return game payoff
	def get_payoff(self, player, game_state):
		#first get the final stack
		final_stack = [p for p in game_state['table'].seats.players if p.uuid == player.uuid][0].stack

		#how do we know the initial stack?

	#flatten a set of valid actions from game state
	#into equivelant of our actions
	def flatten_actions(self, actions, table = None, pot = None, bb = None):
		#actions contains a dictionary like this: [{'action': 'fold', 'amount': 0}, {'action': 'call', 'amount': 10}, {'action': 'raise', 'amount': {'min': 15, 'max': 2015}}]
		#we replace the raise action with our # of specific raise actions
		raise_action = [a for a in actions if a["action"] == "raise"]

		#quit if there is no raise action
		if len(raise_action) == 0:
			#there are no raise actions, nothing to do but return existing actions
			return actions

		#if already flattened, nothing to do
		if raise_action[0].get("abstraction","") != "": return actions

		#figure out the current pot from the table
		if pot == None: pot = self.calculate_current_pot(table)

		#figure out big blind amount
		if bb == None: bb = PokerAbstractor.ABSTRACTED_BB

		#get the raise action, and min/max
		#and remove it from action list
		raise_action = raise_action[0]
		min_raise = raise_action["amount"]["min"]
		max_raise = raise_action["amount"]["max"]
		actions.remove(raise_action)

		#have we shifted max?
		max_shifted = False

		#step through all abstracted raise amounts and add them directly to the action dictionary
		for c in range(PokerAbstractor.ABSTRACTED_RAISES+1):

			#figure out raise amount, ZERO means all-in (max raise)
			if (c == 0): amount = max_raise
			else: amount = bb * 2 ** (2 * c - 1)

			#if the current amount is within 50% of our minimum adjust it up
			#IE: min bet is 60, our bets are 20,80,320 - we disable 20 and keep 80
			#	 min bet is 35 our bets are 20,80,320 - we change 20 to 35
			#	 min bet is 120 our bets are 20,80,320 - we disable 20, shift 80 to 120
			#this makes our bets a little less ridgid
			if amount < min_raise and amount > min_raise // 2: amount = min_raise

			#regardless of the size of the first bet above our max, we shift it down
			#and disable the result
			if not max_shifted and amount > max_raise and c != 0: amount = max_raise

			#if the prior raise is above our max, our all in becomes a call
			if c == 0 and max_raise == -1:

				#add this abstracted action to action dict
				#if amount is within range - NOTE - ALL IN IS ALWAYS VALID EVEN BELOW MIN BET SIZE
				actions.append({
					'action': "call",
					'abstraction': "raise{}".format(c),
					'amount': actions[1]["amount"],
					'valid': True
				})

			else:

				#add this abstracted action to action dict
				#if amount is within range - NOTE - ALL IN IS ALWAYS VALID EVEN BELOW MIN BET SIZE
				actions.append({
					'action': "raise",
					'abstraction': "raise{}".format(c),
					'amount': amount,
					'valid': (amount >= min_raise and amount <= max_raise and min_raise > 0 and max_raise > 0) or c == 0
				})

		#now we have abstracted our raise actions to be just like any other action
		return actions


	#return the list of all game actions (may not be valid for every situation
	def game_actions(self):
		#return the valid actions for the game
		#call and check are treated as the same action
		#raise "ZERO" is all-in
		return []

	#valid actions returns the valid actions for given state
	def valid_actions(self, game, game_state):
		#return the valid actions of the game given game state
		return game.generate_possible_actions(game_state)

	#Kuhn poker has 2 action sets (Bet or Call/Fold), Poker has 4
	def action_sets(self):
		return 2 + PokerAbstractor.ABSTRACTED_RAISES + 1 #fold, call, raise

	#reset a game
	def reset_game(self,game, players):
		#we don't need to reset PyPokerEngine
		pass

	#step the game round
	def step_round(self,game, game_state, abstracted_action):
		#abstracted action contains the action name and amount
		#that was set on the puppet player, we also pass it to update state
		return game.step_round(game_state,abstracted_action[0],abstracted_action[1])
		
	#return if a round is finished or not
	def round_finished(self, game_state):
		#figure out if the round is finished based on game state
		return game_state["street"] == Const.Street.FINISHED

	#step the game until the given player's turn
	def step_to_player(self, game, player, game_state):
		pass

	#display game state
	def display(self, game_state, prefix = ""):
        #gather some basic info from game state
		round = game_state["round"]

        #print that info out
		print(prefix + "Round {}".format(round))

	#assign hole cards
	def assign_hole_cards(self, game_state, player_info, community_cards = []):
		#get the table, then restore and shuffle the deck
		table = game_state["table"]
		table.deck.restore()
		table.deck.shuffle()

		#pop off the community cards so we don't try to assign them to players
		if len(community_cards) > 0:
			[table.deck.deck.remove(c) for c in table.deck.deck if str(c) in community_cards]

		#clear all current hole cards
		#since we are manually assigned them
		#[p.clear_holecard() for p in table.seats.players]

		#assign cards to all players
		for player in table.seats.players:
			#clear curent hole cards
			player.clear_holecard()

			#assign pre-determined cards (or random if not set)
			starting_hole = player_info[player.uuid].get("starting_hole",[])

			#pop those cards from deck so they aren't used elsewhere
			if len(starting_hole) > 0: 
				[table.deck.deck.remove(c) for c in table.deck.deck if str(c) in starting_hole]
				starting_hole = gen_cards(starting_hole)
			else: starting_hole = table.deck.draw_cards(2)

			#set cards for player
			player.hole_card = starting_hole

		#return that game state
		return game_state

	#assign hole cards
	def assign_community_cards(self, game_state, community_card):
		#get the table, then restore and shuffle the deck
		table = game_state["table"]

		#remove current community cards from table
		table_com = table._community_card
		while len(table_com) > 0:

			#remove this card from table and add back to deck
			c = table_com.pop()
			table.deck.deck.append(c)

		#add our community cards to table
		for c in community_card: table.add_community_card(gen_cards([c])[0])

		#remove our community cards from the deck
		[table.deck.deck.remove(c) for c in table.deck.deck if str(c) in community_card]

		#make sure no player has been assigned one of our community cards
		#if so, swap it out with another card from the deck
		for p in table.seats.players:
			for c in p.hole_card:
				if str(c) in community_card:
					p.hole_card.remove(c)
					p.hole_card.append(table.deck.draw_cards(1)[0])

		#return that game state
		return game_state


	#find a set of community cards that matches the given hand profile when paired with the given hole
	def find_community_profile(self, deck, street, hole, target_hand_profile, start_cards = []):
		#our hand profile
		hand_profile = []
		community = []

		#how many community cards, based on street
		community_size = self.street_community_sizes[street]

		#keep going until our hand profile matches the target profile
		while hand_profile != target_hand_profile:

			#get a random set of cards from the table
			community = random.sample(deck,community_size - len(start_cards))

			#abstract the hand info for those cards with our hole
			hand_profile = self.abstract_hand_info(street, hole, start_cards + community)

		#we have found a combination of cards that match our hand profile, return it
		return start_cards + community

	#step the game state to a specific path
	def step_emulator_to_path(self, path, game_state, emulator, players, player_info):

		#this process may be slow:
		print("Stepping Emulator to Path...",end="")

		#we need to track our street as we are going
		#so we know how many cards to add at each street
		street = 0

		#make a reversed copy of our path
		back_path = path.copy()
		back_path.reverse()
		#first entry is always the hole cards and second is always position of player (EP, MP, LP)
		root = back_path.pop()
		pos = back_path.pop()
		
		#update dealer position to match the state where our player will be in position described by path
		#and start a new game state (this will shift everything where we need it
		game_state["table"].dealer_btn = self.unpack_button_pos(pos[0],len(players)) - 1
		game_state, _ = emulator.start_new_round(game_state)

		#update hole of trainer (assuming player 0 always)
		hole_cards = self.unpack_hole(root[0])
		community = []
		player_info[players[0].uuid]["starting_hole"] = [str(hole_cards[0]),str(hole_cards[1])]
		game_state = self.assign_hole_cards(game_state,player_info)

		#continue working until all path items are removed
		#working backwards to forwards (using pop to remove the last item)
		while len(back_path) > 0:

			#remove this path entry
			entry = back_path.pop()

			#if a villian move, setup players and step
			if entry == (1,1):

				#get the next entry, that tells us the move
				entry = back_path.pop()

				#get valid actions and abstract our action name/amount
				valid_actions = emulator.generate_possible_actions(game_state)
				(action_name, action_amount) = self.unpack_action(entry,valid_actions)

				#if action is valid - trigger it
				if action_name != None:

					#first player acts
					game_state, messages = emulator.apply_action(game_state,action_name,action_amount)

				#now whatever that action was (BB, SB, etc) - we need to step to current players return
				#by forcing all other emulators to call

				#calculate call amount for remaining players
				call_amount = emulator.generate_possible_actions(game_state)[1]["amount"]

				#all other players call -> at some point instead of hitting player ZERO
				#we may end up moving to the next street
				while game_state["next_player"] != 0 and game_state["street"] == street:

					#apply call for this player
					game_state, messages = emulator.apply_action(game_state,"call",call_amount)

				#if the street has changed, let's hope the next tuple in our path is the street tuple
				if game_state["street"] != street and back_path[-1] != (3,1):
					#there is an issue - we moved to the next street, but the next path tuple
					#is not a street path - we have gotten out of sync with our path
					print("checking")

				#step to the hero
				#game_state, messages = emulator.step_to_player_turn(game_state,0)

			#if a hero move
			elif entry == (2,1):

				#get the next entry, that tells us the move
				entry = back_path.pop()

				#get action from packed action value (using game state)
				(action_name, action_amount) = self.unpack_action(entry,emulator.generate_possible_actions(game_state))

				#declare our hero action and step round if action is not small/big blind
				if action_name != None: game_state, messages = emulator.apply_action(game_state,action_name, action_amount)

			#if this is a street
			elif entry == (3,1):

				#get the next entry - which describes the cards made from the community
				street += 1
				community_profile = back_path.pop()
				hand_strength = back_path.pop()
				suit_profile = back_path.pop()
				street_profile = [community_profile,hand_strength,suit_profile]

				#now, pick a random set of cards that matches this community profile
				#and add them to the game state (or append to game state)
				community = self.find_community_profile(game_state["table"].deck.deck,street,hole_cards,street_profile, community)

				#replace the game state community cards with these cards
				game_state = self.assign_community_cards(game_state,[str(c) for c in community])

				#here we should make sure to step into the next street within the emulator

		#return that game state based on path
		print("Done!")
		return game_state

	#open monte
	def open_monte(self,shared_name):

		#open existing monte shared memory
		self.monte_shm = shared_memory.SharedMemory(name=shared_name)

		#attach a nupmy array to that shared memoryt
		self.monte_results = np.ndarray(multi_monte_shape(), dtype=np.int32, buffer=self.monte_shm.buf)


	#return pre-calculated monte carlo simulation for the given hole/community
	def get_monte(self, hole, comm):
		#hole and community can be either card objects or strings
		if type(hole[0]) == str: hole = gen_cards(hole)
		if len(comm) > 0 and type(comm[0]) == str: comm = gen_cards(comm)

        #figure out street
		street = len(comm) - 2
		if street < 0: street = 0

		#get total hand profile
		profile = self.hand_profile(hole,comm)

		#get comm profile - default to the street if none is found
		if len(comm) > 0: comm_profile = self.hand_profile(comm[:2],comm[2:])
		else: comm_profile = -1

		#load monte if empty
		if len(self.monte_results) == 0:
			with open('monte.pkl','rb') as fb:
				print("Opening monte.pkl")
				self.monte_results = pickle.load(fb)

		#get rank info for hand
		games = self.monte_results[street][profile][-1][0]
		wins = self.monte_results[street][profile][-1][1]

		#get rank info for community vs hole
		comm_games = self.monte_results[street][profile][comm_profile][0]
		comm_wins = self.monte_results[street][profile][comm_profile][1]

		#get win rates (don't allow divzero error)
		comm_win_rate = 0
		win_rate = 0
		if comm_games > 0: comm_win_rate = comm_wins / comm_games
		if games > 0: win_rate = wins / games

		#if we have community, return that otherwise return calculation from hole
		if comm_games > 0:
			#return those results as a tuple (only Hole vs. Comm results needed)
			return (comm_wins,comm_games,comm_win_rate)
		else:
			#return those results as a tuple (only Hole vs. Comm results needed)
			return (wins,games,win_rate)

	#look through our monte results and find the # of hands that have a higher
	#winning ratio than we do
	def get_montecomm(self, hole, comm):
		#hole and community can be either card objects or strings
		if type(hole[0]) == str: hole = gen_cards(hole)
		if len(comm) > 0 and type(comm[0]) == str: comm = gen_cards(comm)

        #figure out street
		street = len(comm) - 2
		if street < 0: street = 0

		#get hole profile
		profile = self.hand_profile(hole,comm)

		#get comm profile - default to the street if none is found
		if len(comm) > 0: comm_profile = self.hand_profile(comm[:2],comm[2:])
		else: comm_profile = -1

		#load monte if empty
		if len(self.monte_results) == 0:
			with open('monte.pkl','rb') as fb:
				self.monte_results = pickle.load(fb)

		#get rank info for hand
		games = self.monte_results[street][profile][-1][0]
		wins = self.monte_results[street][profile][-1][1]
		win_rate = wins / games

		#step through all hands (besides ours)
		total_hands = len(self.monte_results[street])
		better_hands = 0
		better_games = 0
		better_wins = 0
		for h in range(total_hands):
			if h != profile:
				#get games and wins
				other_games = self.monte_results[street][h][comm_profile][0]
				other_wins = self.monte_results[street][h][comm_profile][1]
				
				#if there were games or wins
				#then calculate other rate and compare to our rate
				if other_wins > 0 and other_games > 0:
					other_rate = other_wins / other_games
					if other_rate > win_rate: 
						print("profile {} rate = {:4.2f}".format(h,other_rate))
						better_hands += 1
						better_games += other_games
						better_wins += other_wins

		#return that
		return (better_hands, total_hands, better_hands / total_hands)
		#return (better_wins,better_games,better_wins / better_games)