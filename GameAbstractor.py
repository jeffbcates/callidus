#a game abstractor is used by the regret manager
#so that we can process regrets by any different type of game
#each game should implement a game abstractor that takes the game
#state for a given timestep of a game and returns a unified regret path
#game state is always a dictionary describing the current game state
class GameAbstractor:

	#initialize game abstractor
	def __init__(self):
		pass

	#get payoff for game
	def get_payoff(self, player, game_state):
		pass

	#display game state
	def display(self, game_state, prefix = ""):
		pass

	#reset a game complete
	def reset_game(self,game, players):
		pass

	#step game round
	def step_round(self,game, game_state):
		pass

	#return if a round is finished or not
	def round_finished(self, game_state):
		pass
			
	#step the game until the given player's turn
	def step_to_player(self, game, player, game_state):
		pass

	#extract and abstract info from the given game state dictionary
	def abstract_game_state(self, game_state, player):
		pass

	#extract an action name and value from a valid action for the game
	def abstract_action(self, action):
		pass

	#return the # of action sets for the game (this should be constant for each game, based on game state abstractor)
	def action_sets(self):
		pass

	#return the list of all game actions (may not be valid for every situation
	def game_actions(self):
		pass

	#return valid actions for current state
	def valid_actions(self, game, game_state):
		pass

class StringAbstractor(GameAbstractor):

	def __init__(self):

		#pass to super - save game reference
		super().__init__()

	#display game state
	def display(self, game_state, prefix = ""):
		pass

	#reset a game complete
	def reset_game(self,game, players):
		pass

	#step game round
	def step_round(self,game, game_state, abstracted_action):
		pass

	#step the game until the given player's turn
	def step_to_player(self, game, player, game_state):
		pass

	def abstract_game_state(self, game_state, player):

		#this is for backwards compatibility with the Tenner Kuhn implementation
		return game_state

	#extract an action name and value from a valid action for the game
	def abstract_action(self, action):
		pass

	#Kuhn poker has 2 action sets (Bet or Call/Fold)
	def action_sets(self):
		return 2

	#return the list of all game actions (may not be valid for every situation
	def game_actions(self):
		return ["B","C"]

	#kuhn poker always has 2 valid actions
	def valid_actions(self):
		return ["B","C"]



class KuhnAbstractor(GameAbstractor):

	def __init__(self):

		#pass to super - save game reference
		super().__init__()

	def abstract_game_state(self, game_state, player):

		#get the card of the current player
		#and return it along with the history
		if game_state["current_player"] == 2:
			return "DONE"
		else:
			return game_state["cards"][game_state["current_player"]] + game_state["actions"]

	#extract an action name and value from a valid action for the game
	def abstract_action(self, action):
		#action is a string
		value = 0
		if action == "B": value = 1
		return action, value

	#return game payoff
	def get_payoff(self, player, game_state):
		return game_state["stacks"][game_state["player_seats"][player.name]] - 2

	#return the list of all game actions (may not be valid for every situation
	def game_actions(self):
		return ["B","C"]

	#valid actions returns the valid actions for given state
	def valid_actions(self, game, game_state):
		#we always have 2 valid actions
		return ["B","C"]

	#Kuhn poker has 2 action sets (Bet or Call/Fold)
	def action_sets(self):
		return 2

	#reset a game
	def reset_game(self,game, players):

		#call reset on the kuhn game
		#and return its results
		return game.reset(players)

	#step the game round
	def step_round(self,game, game_state, abstracted_action):

		#step game round and return game state
		return game.step_round(game_state)

	#return if a round is finished or not
	def round_finished(self, game_state):
		return game_state["round"] == 4 #round 4 is finished for kuhn games

	#step the game until the given player's turn
	def step_to_player(self, game, player, game_state):

		#get the player seat for the given player
		player_seat = game_state["player_seats"][player.name]

		#if the current player is not this player, step the game
		if game_state["current_player"] != player_seat:
			game_state = game.step(game_state)

		#return game state
		return game_state

	#display game state
	def display(self, game_state, prefix = ""):
        #gather some basic info from game state
		pot = game_state["pot"]
		round = game_state["round"]
		cards = game_state["cards"]
		actions = game_state["actions"]

        #print that info out
		print(prefix + "Round {}, Pot: {}, Cards: {}, Actions: {}".format(round,pot,cards,actions))
