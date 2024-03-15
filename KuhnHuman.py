from KuhnPlayer import KuhnPlayer

class KuhnHuman(KuhnPlayer):

    #initialize the player
    def __init__(self, name):
        super().__init__(name)

    #declare action
    def declare_action(self, card, valid_actions, game_state):

        #get the human action:
        action = ""
        while (action not in valid_actions):
            print("P" + str(self.name) + ": You have " + card + ", declare an action " + str(valid_actions) + "?:")
            action = input()

        #set value based on action
        value = 0
        if action == "B": value = 1

        #return results:
        return action, value

    def receive_round_result_message(self, winner, game_state):

        #print who wins
        print("P" + str(winner) + " wins " + str(game_state["payoff"]) + " chips")

        #now determine the winner (utility of each player)
        print(str(game_state))

        #game is done
        print("game over!")

