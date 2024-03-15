#from the old poker engine (will remove soon)
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards

#new poker engine
from pathlib import Path

#used to detect the play area and cards
import cv2
import numpy as np
import glob, os
from matplotlib import pyplot as plt
from shapely.geometry import box
import pyautogui
import random

#constants
card_template_folder = "C:\\users\\jcate\\poker\\cards\\"
monte_simulations = 5000
big_blind = 4

#the prediction engine uses a different format for cards
#this function converts our card format to that format for use in predictions
def convert_hand_format(hand):
    #our output array
    output = []
    
    #step through all cards in the hand
    for i in range(len(hand)):
        suit = hand[i][:1]
        rank = hand[i][-1:]
        if rank == "T": rank = "10"
        elif rank == "J": rank = "11"
        elif rank == "Q": rank = "12"
        elif rank == "K": rank = "13"
        elif rank == "A": rank = "14"
        output.append(rank + "_" + suit)
    #return that hand
    return output



# count opponents
def count_opponents(table_image):

    #convert table image to gray scale
    img_gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)

    #resulting cards
    results = 0
    
    #load opp detection card
    template = cv2.imread(card_template_folder + "OPP.png",0)
            
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

    threshold = 0.95
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        results = results + 1
    #number of cards should be even or zero
    return int(results / 2)



# detect available cards
def identify_table_cards(image):

    #the results of our searches
    #contains only the highest-most ranking and non-overlapping card locations and ranks
    results = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    for file in glob.glob("C:\\users\\jcate\\poker\\cards\\C*.png"):
        #get card name
        card = file.split("\\C")[1].replace(".png","")
        
        #read template
        template = cv2.imread(file)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        found = None
        (tH, tW) = template.shape[:2]
        # cv2.imshow("Template", template)

        #tEdged = cv2.Canny(template, 50, 200)

        for scale in [1.0,1.4210526315789473]: #np.linspace(1, 2, 20):
            # print(card + ": " + str(scale))
            # resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            resized = cv2.resize(gray, dsize = (0,0), fx = scale, fy = scale)

            r = gray.shape[1] / float(resized.shape[1])

            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            #edged = cv2.Canny(resized, 50, 200)
            #result = cv2.matchTemplate(edged, tEdged, cv2.TM_CCOEFF_NORMED)
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:            
                found = (maxVal, maxLoc, r, scale)

        (foundVal, maxLoc, r, scale) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        
        
        #only consider adding to results if match is above a certain amount
        #note: sometimes our routine picks up the logo at the top left of the table screenshot
        #and places it as a third card in our set of hole cards
        #so only consider within a reasonable bounds (inset within the game area)
        if (foundVal > 0.5 and startX > 100 and startY > 100):
            
            #print(card + ": " + str(scale) + " - " + str(int(maxVal * 100)) + " " + str(maxLoc))
        
            #check all current results to see if we intersect one of the results
            #if so - this could be false positive match (or the other one could)
            #we are going to pick the highest scored card for this location
            card_overlaps = 0
            for i in range(len(results)):
                #create a box for both the results and us
                card1 = box(startX,startY,endX,endY)
                card2 = box(results[i][2],results[i][3],results[i][4],results[i][5])

                #if the cards intersect, we either keep the existing
                #card or replace it with ours
                if card1.intersects(card2):

                    #flag as an overlap so we don't add it again
                    #after this loop
                    card_overlaps = 1

                    #if our new card has a better score than existing one
                    #we need to replace it with the existing one
                    #print(suit + rank + " overlaps " + results[i])
                    if (foundVal > results[i][1]):
                        #print(suit + rank + " wins!")
                        results[i] = (card,foundVal,startX,startY,endX,endY,r,scale)


            #if there was not an overlap of cards, add the card
            #the cards do not intersect, add to the end of results
            if card_overlaps == 0: results = results + [(card,foundVal,startX,startY,endX,endY,r,scale)]

    #we return hole cards and community cards
    hole_cards = []
    community_cards = []
                
    #results contains only the highest ranking cards identified
    #at each non-intersecting location
    for i in range(len(results)):
        
        (card,foundVal,startX,startY,endX,endY,r,scale) = results[i]
        color = (255,0,0)
        if (foundVal > 0.5): color = (0,0,0)
        if (foundVal > 0.8): color = (0,255,0)
        

        if (scale == 1.0): 
            cardType = "C"
            community_cards = community_cards + [card]
        else:
            cardType = "H"
            hole_cards = hole_cards + [card]
        
        #cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        #cv2.putText(image,cardType + ":" + card ,(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)


    #plt.imshow(image)
    #print("Community Cards: " + str(community_cards))
    #print("Hole Cards: " + str(hole_cards))
    return hole_cards, community_cards

# Estimate the ratio of winning games given the current state of the game
def estimate_win_rate(nb_simulation, nb_player, hole_card, community_card=None):
    if not community_card: community_card = []

    # Make lists of Card objects out of the list of cards
    community_card = gen_cards(community_card)
    hole_card = gen_cards(hole_card)

    # Estimate the win count by doing a Monte Carlo simulation
    win_count = sum([montecarlo_simulation(nb_player, hole_card, community_card) for _ in range(nb_simulation)])
    return 1.0 * win_count / nb_simulation


#run a single montecarlo simulation
def montecarlo_simulation(nb_player, hole_card, community_card):
    # Do a Monte Carlo simulation given the current state of the game by evaluating the hands
    community_card = _fill_community_card(community_card, used_card=hole_card + community_card)
    unused_cards = _pick_unused_card((nb_player - 1) * 2, hole_card + community_card)
    opponents_hole = [unused_cards[2 * i:2 * i + 2] for i in range(nb_player - 1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0

#run a single montecarlo simulation
def montecarlo_simulation(nb_player, hole_card, community_card):
    # Do a Monte Carlo simulation given the current state of the game by evaluating the hands
    community_card = _fill_community_card(community_card, used_card=hole_card + community_card)
    unused_cards = _pick_unused_card((nb_player - 1) * 2, hole_card + community_card)
    opponents_hole = [unused_cards[2 * i:2 * i + 2] for i in range(nb_player - 1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0

#process a hand
def process_hand(players,win_rate,pot_size,action,hole_card,community_card):

    # call amount is action
    call_amount = action
    amount = None
        
    #calculate pot odds
    call_pot_odds = call_amount / ( call_amount + pot_size )
        
    #calculate max profitable raise and raise pot odds
    max_profitable_raise = (pot_size * ( win_rate - 0.1) ) / (1.1 - win_rate)
        
    #what should we do
    decision = "call"
    amount = call_amount
    
    #should we fold
    if win_rate < call_pot_odds and action > 0: 
        decision = "fold"
        amount = 0
    #jbc: added logic so max raise must be greater than big blind to raise (so we don't raise like "1")
    elif win_rate > call_pot_odds + 0.1 and max_profitable_raise > call_amount and max_profitable_raise > big_blind:
        decision = "raise"
        amount = max_profitable_raise
    
    #print results
    print("win rate: " + str(int(win_rate * 100)) + "% | pot odds: " + str(int(call_pot_odds * 100)) + "% | max raise = " + str(int(max_profitable_raise)))
    print("decision = " + decision + " " + str(int(amount)))

def all_cards():
    
    results = []
    for file in glob.glob("C:\\users\\jcate\\poker\\cards\\C*.png"):
        #get card name
        card = file.split("\\C")[1].replace(".png","")
        results = results + [card]
    
    #return that list
    return results

def card_suit(card):
    
    #convert to numeric
    suit = 1 #clubs
    if card[0] == "D": suit = 2
    elif card[0] == "H": suit = 3
    else: 4
        
    #return that suit
    return suit
    
def card_rank(card):
    
    #convert rank to numeric
    rank = 0
    if card[1] == "T": rank = 10
    elif card[1] == "J": rank = 11
    elif card[1] == "Q": rank = 12
    elif card[1] == "K": rank = 13
    elif card[1] == "A": rank = 14
    else: rank = int(card[1])
    
    #return that rank
    return rank
    
#the following function helps us figure out how the # of players
#affects the win rate of any given random hand
def test_win_rate_for_players():

    #get all cards 
    cards = all_cards()
    
    #deltas start at ZERO
    monte_deltas = [0,0,0,0,0,0,0]
    monte_deltas2 = [0,0,0,0,0,0,0]
    
    #for a very long time...
    for i in range(100 * 1000):
        random.shuffle(cards)
        monte_hand = []
        for i2 in range(8):
            win_rate = estimate_win_rate(1000,2+i2,cards[:2])
            monte_hand.append(win_rate)
            if i2 > 0: 
                monte_deltas[i2-1] = monte_deltas[i2-1] + int((monte_hand[i2-1] - monte_hand[i2]) * 100)
                monte_deltas2[i2-1] = monte_deltas2[i2-1] + int((monte_hand[i2] / monte_hand[0]) * 100)
        delta_avg = [0,0,0,0,0,0,0]
        delta_avg2 = [0,0,0,0,0,0,0]
        if (i+1) % 100 == 0:
            for i2 in range(7):
                delta_avg[i2] = round(monte_deltas[i2] / (i+1),2)
                delta_avg2[i2] = round(monte_deltas2[i2] / (i+1),2)
            #print("sim." + str(i+1) + " a1: " + str(delta_avg))
            print("sim." + str(i+1) + ": " + str(delta_avg2))
    #we are done
    print("completed running simulations...")
    
#main entry
version = 1.0
if __name__ == '__main__':
        #give some instructions
        print("Welcome to Callidus Interactive " + str(version))
        
        #load the probabilty model
        #print("Loading Probability Model...")
        #probabilityModel = load_monte_model()
        
        #some basic vars
        hole_cards = []
        community_cards = []
        pot = 0
        action = 0
        table = 0
        players = 2
        screenshots = 0
        win_rate = 0
        chips = 0 #how many chips we have in the hand (sum of various actions)
        
        #continue until command is quit
        command = ""
        while command != "quit":
            #get next command from user
            command = input()
            
            #split commands by space
            commands = command.split(" ",1)
            
            #based on first part of command, do something cool
            if commands[0] == "?":
                print("p: set pot")
                print("h: set or clear player hand")
                print("c: set, clear, or add to community hand")
                print("a: action to player, with amount")
                print("s: current game stats | screen capture")
                print("i: set number of players in game")
                print("r: reset the game")
                print("t: add to the table money")
                print("z: screenshot")
                print("?: this lame list")
            if commands[0] == "s":
                #print some stats
                print("~~current hand~~")
                print("players: " + str(players))
                print("hole cards: " + str(hole_cards))
                print("community cards: " + str(community_cards))
                print("pot: " + str(pot))
                print("table: " + str(table))
                print("action: " + str(action))
            if commands[0] == "t" and len(commands) == 1:
                table = 0
                print("cleared table")
            if commands[0] == "t" and len(commands) > 1:
                table = table + int(commands[1])
                print("table is now " + str(table))
            if commands[0] == "r":
                pot = 0
                hole_cards = []
                community_cards = []
                action = 0
                table = 0
                print("reset game")
            if commands[0] == "i":
                players = int(commands[1])
                print("players now " + str(players))
            if commands[0] == "p":
                pot = int(commands[1])
                print("pot now " + str(pot))
            if commands[0] == "h" and len(commands) == 1:
                hole_cards = []
                table = 0
                print("hole cards and table cleared")
            if commands[0] == "h" and len(commands) > 1:
                hole_cards = commands[1].upper().split(" ")
                table = 0 
                print("hole cards now " + str(hole_cards) + " and table cleared")
            if commands[0] == "t":
            
                #print predicted monte
                #p = predict_win_rate(probabilityModel,players,hole_cards,community_cards)
                #print("predicted monte: " + str(int(p*100)) + "%")
            
                #test several different montes on current cards, does it make a diff?
                win_rate = estimate_win_rate(500, players, hole_cards, community_cards)
                print("actual monte: " + str(int(win_rate * 100)))
                
                #now blow it out with a real long test
                #print("running monte 5k...")
                #w = estimate_win_rate(5000, players, hole_cards, community_cards)
                #print("monte 5k = " + str(int(w * 100)) + "%")
            
            if commands[0] == "z" and len(commands) == 1:
                #let user know what we are doing
                print("reading table...")
            
                #take a screenshot and save it both as "table" and as screenshot
                screenshot = pyautogui.screenshot(region=(0,0,900,600))
                screenshot.save(r'C:\users\jcate\poker\screenshots\table.png')
                
                #reload using cv2
                img_rgb = cv2.imread(r'C:\users\jcate\poker\screenshots\table.png')
                screenshots = screenshots + 1
                
                #count the # of opponents using that screenshot
                players = count_opponents(img_rgb) + 1
                print("players identified: " + str(players))
                
                #now, use that screenshot to determine community cards and hole cards
                community_cards = []
                hole_cards = []
                hole_cards, community_cards = identify_table_cards(img_rgb)
                table = 0
                print("hole identified: " + str(hole_cards))
                print("community identified: " + str(community_cards))
                
                #only if we have 2 hole cards
                if len(hole_cards) == 2:
                
                    #now calculate the win rate
                    #win_rate = predict_win_rate(probabilityModel, players, hole_cards, community_cards)
                    #print("predicted win rate: " + str(int(win_rate * 100)) + "%")
                    win_rate = estimate_win_rate(500, players, hole_cards, community_cards)
                    print("actual win rate: " + str(int(win_rate * 100)) + "%")
                    
                else:
                
                    print("could not identify hole cards!")
            
            if commands[0] == "c" and len(commands) == 1:
                community_cards = []
                table = 0
                win_rate = 0
                print("community cards and table now cleared")
            if commands[0] == "c" and len(commands) > 1:

                #get community cards input
                input_cards = commands[1].upper().split(" ")
                
                #if 3 cards are passed this is a new hand
                if len(input_cards) == 3:
                    community_cards = input_cards
                else:
                    community_cards = community_cards + input_cards
                
                #print out the results and clear table
                table = 0
                win_rate = 0
                print("community cards now: " + str(community_cards) + " | table cleared")
            if commands[0] == "a":
            
                #if win rate zero, calc now
                if (win_rate == 0):
                    print("calculating win rate...")
                    win_rate = estimate_win_rate(500, players, hole_cards, community_cards)
                    print("win rate is " + str(int(win_rate * 100)) + "%")
                    

            
                #action command, the current call amount is set
                if len(commands) == 1: action = 0
                else: action = int(commands[1])
                
                #confirm
                print("~~action~~")
                print("h " + str(hole_cards) + " | c " + str(community_cards) + " with " + str(players) + " players")
                print("pot " + str(pot) + " + table " + str(table) + " = " + str(pot+table)) 
                print("action is " + str(action))
                
                #run simulations
                process_hand(players,win_rate, pot + table,action,hole_cards,community_cards)
                
            
            if commands[0] == "w":
                print("Testing win rates for players...")
                test_win_rate_for_players()