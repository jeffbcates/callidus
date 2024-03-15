#REGRET BROWSER COMMANDS
import sys
import pickle
import numpy as np
import time
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing import Process
from callidus import EmulatorPlayer
from Regrets import InformationSet, RegretManager

#helpful utilities
from util import estimate_prob, progressBar, setup_emulator_players, multi_monte, multi_monte_shape
from ai_caller import FishPlayer
from ai_minraiser import RaisePlayer
from ai_monte import DataBloggerBot
from InteractiveHuman import InteractiveHuman

#for pypoker engine
from pypokerengine.api.game import start_poker, setup_config
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.hand_evaluator import HandEvaluator as he
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards,estimate_hole_card_win_rate

class BrowserCommands:

    #initialize commands
    def __init__(self, browser, regret_man):

        #save browser reference
        self.browser = browser
        self.regret_man = regret_man
        self.bigblind = 10 #default
        self.identity = None #defualt to no identity here
        self.monte_results = [] #where we store monte simulation results
        self.trace_path = [] #trace path
        self.catch_errors = True #catch all errors from system
        self.use_monte = True
        self.use_cfr = True
        self.use_argmax = False
        self.regret_ondisk = False #are our regrets on-disk

    #register all commands
    def register(self):

        #register all our commands with passed browser
        self.browser.register_command("exit","Exit the regret browser",self.exit,0)
        self.browser.register_command("cd","Change current dictionary",self.cd,1,
            [   "CD [PATH | .. | \\]","",
                "  PATH The relative path of a dictionary to current dictionary",
                "  .. changes to the parent dictionary",
                "  \\ changes to the root dictionary"
            ])
        self.browser.register_command("ls","List regrets in current dictionary",self.ls,0,
            [   " LS [R] [DEPTH]","",
                "  R recursively list all paths",
                "  DEPTH numeric depth to recurse"
            ])
        self.browser.register_command("help","Display command help",self.help,0,
            [   "HELP [COMMAND]","",
                "  COMMAND - displays details about the command, parameters","",
                "  If no parameters are provided, lists all available commands"
            ])
        self.browser.register_command("mem","Manipulate shared training memory",self.mem,0,
            [   "MEM [A | ATTACH | R | READ | W | WRITE | S | START | F | FREE] [IDENTITY] [VALUE]","",
                "  R | READ - reads current shared memory (default parameter)","",
                "  W | WRITE - writes shared memory (must include [IDENTITY] and [VALUE] parameters)",
                "  S | START - starts the shared memory service using this browser",
                "  A | ATTACH - attaches to existing shared memory service on another browser",
                "  F | FREE - closes the shared memory service and frees shared memory buffers",
                "  IDENTITY - the numeric identity (0..7) of the shared memory register to WRITE",
                "  VALUE - the string value to WRITE to shared memory at specified IDENTITY","",
                "  Note: Shared Memory Services OR memory mapped regret files (OPEN / CREATE) are required to run multi-core training!"
            ])
        self.browser.register_command("eval","Evaluates training for the current node",self.eval,0)
        self.browser.register_command("abb","Abstracts a bet amount by current big blind",self.abb,1,
            [   "ABB AMOUNT","",
                "  AMOUNT - The numeric value to abstract using xBB multipler","",
                "  Note: You must call SETBB (once) before calling ABB"
            ])
        self.browser.register_command("transp","Translate path from hole/community cards",self.transp,2,
            [   "TRANSP CARD CARD [CARD] [CARD] [CARD]","",
                "  CARD - in format {suit}{rank}, i.e. H5 HA","",
                "  SUITS - H,C,D,S",
                "  RANKS - 2,3,4,5,6,7,8,9,T,J,Q,K,A","",
                "  Note: The first two cards are not optional, the others are"
            ])
        self.browser.register_command("load","Loads the regret manager from disk to memory(optionally reloads a specific root path)",self.load,0,
            [   "LOAD [PATH]","",
                "  [PATH] - The root regret path to reload","",
                "  Note: [PATH] is optional and if not provided, all root paths are reloaded"
            ])
        self.browser.register_command("create","Creates the regret manager on disk (overwriting existing files)",self.create,0,
            [   "CREATE","",
                "  Note: All contents of existing regrets will be erased"
            ])
        self.browser.register_command("open","Opens the regret file for memory-mapped access",self.open,0,
            [   "OPEN","",
                "  Note: Regrets are opened as memory-mapped and all data operations are performed on disk (no need to save)"
            ])
        self.browser.register_command("simulate","Runs a simulation on current regrets",self.simulate,1,
            [   "SIMULATION ITERATIONS","",
                "  [ITERATIONS] - The # of iterations to run"
            ])
        self.browser.register_command("strain","Trains the current regret model",self.train,1,
            [   "TRAIN IDENTITY MINUTES","",
                "  IDENTITY - The shared identity for this instance of the trainer",
                "  MINUTES - How many minutes to train"
            ])
        self.browser.register_command("train","Trains the current regret model on multiple cores",self.multicoretrain,1,
            [   "TRAIN MINUTES CORES","",
                "  CORES - The number of cores to run on",
                "  MINUTES - How many minutes to train"
            ])
        self.browser.register_command("hole","Sets current hole card and clears community cards",self.hole,2,
            [   "HOLE CARD CARD","",
                "  CARD - in format {suit}{rank}, i.e. H5 HA","",
                "  SUITS - H,C,D,S",
                "  RANKS - 2,3,4,5,6,7,8,9,T,J,Q,K,A"
            ])
        self.browser.register_command("comm","Sets or adds to current community cards",self.comm,1,
            [   "COMM CARD [CARD] [CARD] [CARD] [CARD]","",
                "  CARD - in format {suit}{rank}, i.e. H5 HA","",
                "  SUITS - H,C,D,S",
                "  RANKS - 2,3,4,5,6,7,8,9,T,J,Q,K,A","",
                "  Note: The first card is required, others are optional"
            ])
        self.browser.register_command("prob","Display hand probabilities",self.show_probs,1,
            [   "PROB RANGEMAX","",
                "  RANGEMAX - rangemax value to normalize probabilities"
            ])
        self.browser.register_command("play","Plays a game against Callidus",self.play,0,
            [   "ACT AMOUNT","",
                "  AMOUNT - the amount of current action","",
                "  Note: This will abstract the action amount and search the curent dictionary for a corresponding bet."
            ])
        self.browser.register_command("save","Saves current regrets",self.save,0)
        self.browser.register_command("monte","Run monte simulation",self.monte,1,
            [   "MONTE [BUILD | SHOW | CARD ... ] [ITERATIONS]","",
                "  ITERATIONS - the number of iterations to run"
            ])
        self.browser.register_command("dumplog","Dump regret logs to file",self.dumplog,0)
        self.browser.register_command("reset","Reset all regrets under current path",self.reset,0)
        self.browser.register_command("trace","Sets current path as trace path for training", self.trace,0)
        self.browser.register_command("convert","ONE TIME USE - convert from packed types to separate type arrays", self.convert,0)
        self.browser.register_command("set","Sets a variable value.",self.set,2,
            [   "SET VARIABLE VALUE","",
                "VARIABLE - the name of the variable to set",
                "VALUE - the value to set the variable","","",
                "USEMONTE - turn on/off monte carlo simulation during live play (default true)",
                "USECFR - turn on/off regret minimization during live play (default True)",
                "BB - sets the bigblind value (default 10)",
                "ID - sets the shared identity of this session",
                "NOCATCH - turns on/off error catching (turn off to debug)",
                "ARGMAX - turns on/off argmax: on picks top strategy, off picks weighted random choice"
            ])
        self.browser.register_command("get","Gets a variable value.",self.get,1,
            [   "VARIABLE - the name of the variable to get.  See set command for list of variables."
            ])
        self.browser.register_command("np","Test nump file operations",self.numpytest,2,
            [   "NP [ OPEN | SET | GET | TEST] [FILE | INDEX | ID] [VALUE] "])

    #test numpy operations in multi-processing setting
    def numpytest(self,parameters):

        #create a new file
        if parameters[0] == "CREATE":

            #filename is second param
            filename = parameters[1]

            #create the file
            self.numpy_test_array = np.lib.format.open_memmap(filename,mode="w+",dtype=np.uint32, shape=(1000,))
            print("Created file {} as a memory mapped array".format(parameters[1]))

        #open a file
        if parameters[0] == "OPEN":

            #filename is second param
            filename = parameters[1]

            #open numpy for read/write
            self.numpy_test_array = np.load(filename,mmap_mode ='r+',allow_pickle = False,fix_imports = False)
            print("Opened file {} as memory mapped array".format(parameters[1]))

        #set a value
        if parameters[0] == "SET":

            #set a value on the array
            self.numpy_test_array[int(parameters[1])] = int(parameters[2])
            print("Set value at index {} to {}".format(parameters[1],parameters[2]))

        #get a value
        if parameters[0] == "GET":

            #get a value
            print("Value at index {} = {}".format(parameters[1],self.numpy_test_array[int(parameters[1])]))

        #perform a test
        if parameters[0] == "TEST":

            #let users know
            print("Beginning a test of multi-process file access")

            #our identity is next parameter
            identity = int(parameters[1])
            iterations = 1000

            #for some number of iterations
            for i in range(iterations):

                #progress
                progressBar("Testing",i,iterations)

                #if the value at index 0 does not equal us
                #update to us and wait until it does again
                if self.numpy_test_array[0] != identity:

                    #set the value
                    self.numpy_test_array[0] = identity

                    #wait until it doesn't equal us
                    while self.numpy_test_array[0] == identity:

                        nothing_var = 1

                        #just wait a bit
                        #time.sleep(0.001)

            #we are done testing
            print("")
            print("Testing completed")

    #convert -> extract type information from all values in all arrays and write those values to their type arrays
    def convert(self,parameters):

        #let user know
        print("Converting packed types to type arrays.  This may take a very long time")

        #get our symm tree directly
        symm = self.regret_man.symm_tree

        #step through all arrays in the symm tree
        for ax in range(len(symm._arrays)):
            #get information for this level
            (shape,dtype,total,used) = symm.levelinfo(ax)

            #only do conversion for unint type arrays
            if dtype == 0:

                #print some useful info
                print("Converting level {} ...".format(ax))

                #get reference to array and type array
                t = symm._types[ax]
                a = symm._arrays[ax]

                #calculate length of arrays
                length = shape * used

                #copy from original array into type array (as byte)
                t[:length] = np.mod(a[:length],symm._shape[-1]).astype(np.uint8)

                #strip the types from the original array
                a[:length] //= symm._shape[-1]

            else:
                #let them know
                print("Skipping level {} which is not packed".format(ax))
            


    #help command
    def help(self,parameters):

        #if a command is defined, show details
        command = ""
        if len(parameters) > 0: command = parameters[0]

        #if a command is defined and found
        command = self.browser.commands.get(command,None)
        if command != None:

            #print help for command
            #and all details
            print(command["help"])
            print("")
            details = False
            for d in command["details"]:
                print(d)
                details = True

            #that was the specifics
            if details: print("")
            return

        #step through all commands and print their help
        print("Available Commands:")
        print("Type help [command] to see more details about a specific command")
        for c in self.browser.commands:
            print(c + " " * (15-len(c)) + self.browser.commands[c]["help"])


    #dump regret log to file
    def dumplog(self,parameters):
        print("Dumping Regret Logs...")
        with open('regret_paths.log','wb') as fb:
            pickle.dump(self.regret_man.regret_log, fb)


    #exit the browser
    def exit(self,parameters):

        #let the user know what's happening
        print("Shutting Down Browser")

    #change dictionary of regrets
    def cd(self,parameters):

        #get path and node references
        path = self.browser.current_path
        node = self.browser.current_node

        #check for special CD parameters
        if parameters[0] == ".." and len(path) > 0: path.pop(-1)
        elif parameters[0] == "\\": path.clear()
        else:

            #the path could contain \ separating multiple dictionaries
            subpaths = parameters[0].split("\\")

            #attach those paths to current path
            for subpath in subpaths: 

                #move to this path
                path.append(subpath)

                #check that new path is valid
                if self.browser.gen_current_node(path) == None:
                    print("Dictionary {} Not Found".format(subpath))
                    path.pop()
                    return


    #memory commands
    def mem(self, parameters):

        #default is "R"
        if len(parameters) == 0: parameters.append("A")

        #start shared memory
        if parameters[0] in ["S","START"]:

            #start up shared memory and save them in our regret manager
            #the first entry in our list is not a regret - it triggers shared training
            self.regret_man.regret_locks = shared_memory.ShareableList([" " * 17] * 16, name="REGRET_LOCKS")
            return

        #free shared memory
        if parameters[0] in ["F","FREE"]:

            #let user know what we are doing
            print("Freeing shared memory services for *ALL* running instances!")
            self.regret_man.regret_locks.shm.unlink()
            self.regret_man.regret_locks.shm.close()
            self.regret_man.regret_locks = None
            return

        #for read and write, start shared memory if needed
        #locate shared memory
        if self.regret_man.regret_locks == []:
            self.regret_man.regret_locks = shared_memory.ShareableList(name="REGRET_LOCKS")

        #if attach, quit letting user know we are attached
        if parameters[0] in ["A","ATTACH"]:
            print("Attached to Shared Memory!")
            return

        #write to memory
        if parameters[0] in ["W","WRITE"]:

            #write to shared memory at identity
            lock_identity = int(parameters[1])
            self.regret_man.regret_locks[lock_identity] = parameters[2]

        #if we got this far we need to read memory
        for c in range(len(self.regret_man.regret_locks)):
            print("LOCK {}: {}".format(c,self.regret_man.regret_locks[c]))

    #reset current path and all paths below it
    def reset(self,parameters, path = [], depth = 0):

        #if we are reseting the entire tree, make sure
        if depth == 0 and path == [] and self.browser.current_node == None:
            print("This will reset the entire Symmetric Tree.  Are you sure?  ",end="")
            confirmation = input().upper()[0]
            if confirmation == "Y":

                #we need to write this somehow
                print("To be implemented, LOL")
                return

            else:

                #this was a mistake
                print("Okay, I won't destroy the tree now.  If you change your mind, let me know.")
                return


        #show banner for first level only
        if depth == 0: print("Reseting Path...")

        #shortcut ref to symm tree
        symm = self.regret_man.symm_tree

        #if first depth, get current node
        if len(path) == 0 and self.browser.current_node != None: path = self.browser.current_node

        #unpack our location
        child_value, starting_index, children_length, children_shape = symm.unpack(path)

        #cannot do this using recursion
        indexes = [0]
        lengths = [children_length]
        done = False
        move_next = False
        push_child = False
        while not done:

            #infosets currently exist as only children
            if children_shape == self.regret_man.INFOSET_PATH:

                #reset read and write values
                symm.set(path + RegretManager.PATH_INFOSET_STAT,[0,0])

                #reset regrets to zero
                action_sets = self.regret_man.game_abstractor.action_sets()
                symm.set(path + RegretManager.PATH_INFOSET_REGRETS,[0 for i in range(action_sets)])

                #reset strategy to default
                symm.set(path + RegretManager.PATH_INFOSET_STRAT,[1/action_sets for i in range(action_sets)])

                #pop out of this child
                indexes.pop()
                lengths.pop()
                path.pop()

                #now move to the next child
                move_next = True
                push_child = True

            #this item has children
            else:

                #are we done with this items children?
                if indexes[-1] >= lengths[-1]:

                    #we need to step out of this path
                    indexes.pop()
                    lengths.pop()
                    path.pop()

                    #now move to the next child
                    move_next = True
                    push_child = True

                #we are not done with this items children
                #so we need to push the child, but not move next
                else:

                    #push the child
                    push_child = True

            #if the array is now empty we are done
            if len(indexes) == 0:

                #we are done
                done = True
                move_next = False
                push_child = False

            #are we moving to the  next child?
            if move_next:

                #increment index
                indexes[-1] += 1
                move_next = False

            #do we need to push the current child onto the stack
            if push_child:

                #done pushing child
                push_child = False

                #lookup the next child
                child = symm.child(path, indexes[-1])

                #if this index of child is not found, do nothing, bug flag for "move next"
                if child == None:

                    #clear children shape in case itwas INFOSET before
                    children_shape = 0
                    move_next = True

                else:

                    #extract values from child tuple
                    (parent_index,parent_type, child_value, starting_index, children_length, children_shape) = child

                    #add to path
                    path += [(parent_index, parent_type)]

                    #push onto our stack
                    indexes.append(0)
                    lengths.append(children_length)


    
    #list current dictionary contents
    def ls(self,parameters, path = [], depth = 0):

        #show banner for first level only
        if depth == 0: print("Current Regrets at Path: ")

        #quit if past max depth
        if len(parameters) > 1:
            if depth > int(parameters[1]):
                return

        #if first depth, get current node
        if len(path) == 0 and self.browser.current_node != None: path = self.browser.current_node

        #unpack our location
        _,children_shape = self.regret_man.symm_tree.get(path, include_type=True)

        #infosets currently exist as only children
        if children_shape == self.regret_man.INFOSET_PATH:
            infoset = InformationSet(path,self.regret_man.symm_tree,False)
            print("\n" + " " * depth * 2 + "Information Set Contents: ")
            print(" " * depth * 2 + "{} Reads {} Writes".format(infoset.reads(),infoset.writes()))
            print(" " * depth * 2 + "Average Strategy: " + str(infoset.get_average_strategy()))
            print(" " * depth * 2 + "Strat Sum: " + str(infoset.strategy()))
            print(" " * depth * 2 + "Regrets: " + str(infoset.regrets()))
            print("")

        #if this is not an infoset path, show children
        else:
            #list all children
            for child in self.regret_man.symm_tree.children(path):
                #get display name of node
                display_name = self.regret_man.game_abstractor.unpack_path_tuple([child])

                #print node name and regrets
                print(" " * depth * 2 + str(display_name[0]) + " " + str(child) + ": "  + " Regrets ")

                #if recursive - print grandchildren
                if len(parameters) > 0:
                    if parameters[0] == "R":
                        self.ls(parameters,path + [(child[0],child[1])],depth + 1)

    #evaluate training for current node
    def eval(self, parameters):

        #eval the node and print info
        (regrets,strategy,trained) = self.regret_man.eval_training(False)
        print("Training Review of Model:")
        print("Regret Nodes: {}".format(regrets))
        print("Strategy Nodes: {} | Trained: {}".format(strategy,trained))
        print("")

        #if in "quick" mode, return immediately - this next part is quite slow
        if len(parameters) > 0: return

        #get size info for the tree and print it out here
        size = self.regret_man.symm_tree.size()        
        for sx in range(len(size)):
            #calculate level sparsity
            density = self.regret_man.symm_tree.leveldensity(sx)

            #summarize sparsity
            if float(size[sx]["used"]) > 0:
                total_density = float(density[-1]) / float(size[sx]["used"]) / float(size[sx]["shape"]) * 100
            else:
                total_density = 0

            #print that info
            print("LEVEL {}: {} wide, {}% filled: {}% density :: {} entries - {} avail".format(
                str(sx).rjust(2),
                str(size[sx]["shape"]).rjust(4),
                "{:4.2f}".format(size[sx]["used"] / size[sx]["total"] * 100).rjust(6),
                "{:4.2f}".format(total_density).rjust(6),
                size[sx]["used"],
                size[sx]["total"] - size[sx]["used"]
            ))


    #set big blind
    def setbb(self, parameters):

        #set big blind for path translations
        self.bigblind = float(parameters[0])

        #let user know
        print("Big Blind Set to " + str (self.bigblind))

    #abstract amount by big blind
    def abb(self, parameters):

        #make sure BB is set
        if self.bigblind == 0:
            print("Set BB first by using setbb command!")
        else:

            #get the bet amount
            betamt = float(parameters[0])

            #abstract amount using abstractor
            abb = self.regret_man.game_abstractor.abstract_amount(betamt, self.bigblind)
            print("ABB = " + str(abb))

    #translate cards into a path
    def transp(self, parameters):

        #the first two cards make the hole
        #the rest make community card
        hole_card = parameters[0:2]
        community_card = parameters[2:]
                
        #determine street from those lengths
        street = 0
        if len(community_card) == 3: street = 1
        if len(community_card) == 4: street = 2
        if len(community_card) == 5: street = 3

        #get abstractor reference
        abstractor = self.regret_man.game_abstractor

        #now get the hand info from all that info
        hand_info = abstractor.abstract_hand_info(street,abstractor.gen_cards(hole_card),abstractor.gen_cards(community_card))
        print("Path: " + str(hand_info))

    #update current hole card, this also updates current path
    def hole(self, parameters):

        #get abstractor reference
        abstractor = self.regret_man.game_abstractor

        #the first two cards make the hole
        #the rest make community card
        hole_card = parameters[0:2]
        self.hole_card = abstractor.gen_cards(hole_card)
        self.community_card = []
        self.street = 0
                
        #now get the hand info from all that info
        hand_info = abstractor.abstract_hand_info(self.street,abstractor.gen_cards(hole_card),[])
        self.browser.current_path = [hand_info]

    #update current hole card, this also updates current path
    def comm(self, parameters):
        print("COMM - Not Yet Implemented!")

    #create regret manager on disk
    def create(self, parameters):

        #if there is a parmater its the name on disk
        if len(parameters) > 0:

            #regrets are now on disk
            self.regret_ondisk = True
        
            #load regret manager
            print("Building Regrets on Disk, This May Take a Moment!...")
            self.regret_man.create(parameters[0].lower())
            print("")

        else:

            #we just create in memory
            self.regret_man.create(None)

        

    #open regret manager on disk
    def open(self, parameters):

        #regrets are now on disk
        self.regret_ondisk = True

        #get filename (assume lower case)
        filename = parameters[0].lower()

        #load regret manager
        print("Opening {} on Disk... BRB".format(filename))
        self.regret_man.open(filename)
        print("")

    #load the regret manager (optionally from a path)
    def load(self, parameters):

        #load regret manager
        self.regret_man.load(True)
        print("")

    #save the regret manager (optionally from a path)
    def save(self, parameters):

        #save each regret path
        self.regret_man.save(True)
        print("")

    #trace current path when training / etc
    def trace(self, parmaeters):

        #let user know
        print("Trace path is now: " + "\\".join(self.browser.current_path))
        self.trace_path = self.browser.current_node

    #train current regrets
    def train(self,parameters):

        #add our identity if set already
        if self.identity != None: parameters.insert(0,self.identity)

        #what shared identity and what training iterations
        training_games = 0
        shared_identity = int(parameters[0])
        if len(parameters) > 1 : training_timelimit = int(parameters[1]) * 60
        else: training_timelimit = 0

        #do we have starting cards defined?
        starting_cards = []
        if len(parameters) >= 4: starting_cards = [parameters[2:]]

        #setup a new bot using our current regrets
        #set our shared identity
        cfr_bot = EmulatorPlayer(self.regret_man)
        cfr_bot.regret_man.set_shared_identity(shared_identity)

        #set trace path
        cfr_bot.trace_path = self.trace_path

        #verbosity based on starting cards
        verbosity = 2
        if len(starting_cards) > 0: verbosity = 1

        #start the training
        print("Training Regrets With Identity {}".format(shared_identity))
        cfr_bot.train(training_games,starting_cards,training_timelimit = training_timelimit, verbosity=verbosity, training_path=self.browser.current_node)

    #train current regrets
    def multicoretrain(self,parameters):

        #first parmaeter is minutes
        minutes = int(parameters[0]) * 60

        #second parameter is cores
        if len(parameters) > 1: cores = int(parameters[1])
        else: cores = multiprocessing.cpu_count()
        
        #let the user know what we are doing
        print("\n\nStarting Training Session For {} Minutes on {} Cores".format(minutes,cores))
        print("ips = Iterations Per Second (Single Tree Iteration), sps = Steps Per Second (Complete Tree Traversal)\n".format(minutes,cores))

        #we need to setup a shared memory array that all processes can use to report their status back to us
        #we will provide 10 integer registers that the individual trainers can write their status
        status_size = 5
        training_status = shared_memory.ShareableList([0] * (status_size * cores + 1), name="MULTI_TRAINING_STATUS")

        #the last value of array stores status size to help out trainers
        training_status[-1] = status_size
        
        #figure out regret shared space
        shared_space = self.regret_man.shared_space
        on_disk = self.regret_man.on_disk

        #configure our processes for the # of cores
        processes = []
        for c in range(cores):
            #track our progress
            progressBar("Attaching {} Cores".format(cores),c,cores)

            #setup this process
            p = Process(target=EmulatorPlayer.paralleltrainer, args=(c,status_size,"MULTI_TRAINING_STATUS",shared_space,on_disk,minutes,self.use_argmax, self.use_cfr, self.use_monte, self.regret_man.relative_size))
            p.start()
            processes.append(p)

        #now run through our training, updating our total status every so often
        any_alive = True
        prior_epoch = 0
        timetaken = time.time()
        while any_alive:
            #structure of training status:
            #
            #   EPOCH , STEP , EPOCH_SIZE , REGRETS , TIME
            #



            #get minimum epoch, sum of trainers at that epoch and # of trainers at that epcoh
            min_epoch = min([training_status[i] for i in range(0,len(training_status),status_size)])
            min_trainers = len([training_status[i] for i in range(0,len(training_status),status_size) if training_status[i] == min_epoch])
            epoch_steps = sum([training_status[i] for i in range(1,len(training_status),status_size) if training_status[i-1] == min_epoch])
            epoch_iterations = sum([training_status[i] for i in range(3,len(training_status),status_size)])
            epoch_time = max([training_status[i] for i in range(4,len(training_status),status_size)])
            epoch_speed = 0
            epoch_step_speed = 0
            

            #all trainers that are not on the min epoch will be considered 100% for that epoch
            epoch_steps += (cores - min_trainers) * 100

            #calculate speed if possible
            if epoch_time > 0: 
                epoch_speed = epoch_iterations / epoch_time
                epoch_step_speed = epoch_steps / epoch_time

            #if we are on a new epoch, move to the next line
            if prior_epoch < min_epoch:
                if prior_epoch > 0: print("")
                prior_epoch = min_epoch

            #now we can show the details
            #print(training_status)
            progressBar("Epoch {}".format(min_epoch),epoch_steps,cores * 100,"{:4.2f} sps {:4.1f} ips {:4.1f} sec".format(epoch_step_speed,epoch_speed,epoch_time))

            #wait a few seconds
            time.sleep(2)

            #are any alive
            any_alive = sum([1 for b in processes if b.is_alive() == True])

        #the training has completed, we need to clean up after ourselves
        timetaken = time.time() - timetaken
        print("\n\nTraining Complete after {:4.2f} Seconds.  Cleaning up Processes...".format(timetaken))
        for p in processes:
            p.close()
           

    #run a simulation using current regret manager
    def simulate(self,parameters):

        #setup a new bot using our current regrets
        cfr_bot = EmulatorPlayer(self.regret_man)

        #run some simulations in any case
        print("Running Simulation:")
        #cfr_bot.regret_man.load()
        cfr_bot.allow_retraining = False #we can retrain as needed during simulations
        cfr_bot.training_timelimit = 15 #we simulate for 15 seconds while retraining

        stack_log = []
        game_result = None
        cfr_bot.training = False
        cfr_bot.use_argmax = self.use_argmax
        cfr_bot.use_cfr = self.use_cfr
        cfr_bot.use_monte = self.use_monte
        wins = 0
        games = int(parameters[0])
        winnings = 0

        #setup configuration
        config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="p1", algorithm=cfr_bot)
        for i in range (5):
            #config.register_player(name="p{}".format(i+2), algorithm=FishPlayer())
            config.register_player(name="p{}".format(i+2), algorithm=DataBloggerBot(abstractor = self.regret_man.game_abstractor))

        
        #start progress bar right away so we know something is happening
        progressBar("Progress",0,games,"{} wins {} loses > {:4.2f} bb/100     ".format(0,0,0))
        for round in range(games):
            game_result = start_poker(config, verbose=0)
            progressBar("Progress",round,games,"{} wins {} loses > {:4.2f} bb/100     ".format(wins,round+1-wins,(winnings-round*1000) / 10 / (cfr_bot.rounds+1) *100))
        
            outcome = [player['stack'] for player in game_result['players'] if player['uuid'] == cfr_bot.uuid][0]
            stack_log.append(outcome)
            winnings += outcome

            if outcome > 1000: wins += 1

            progressBar("Progress",round,games,"{} wins {} loses > {:4.2f} bb/100     ".format(wins,round+1-wins,(winnings-round*1000) / 10 / cfr_bot.rounds *100))

        #final outcome numbers
        print("")
        print("============================")
        print("Outcome of {} Games:".format(games))
        print("Hands Played: {}".format(cfr_bot.rounds))
        print("Win Rate: {:4.2f} bb/100".format( (winnings-games*1000) / 10 / cfr_bot.rounds *100))
        print("Wins: {}".format(wins))
        print("Loses: {}".format(games-wins))
        print("Winnings: {}".format(winnings - games * 1000))
        print("Nash: {}".format((games * 1000 - winnings) / (games * 1000) ))
    
    #play the game using an emulator to control the game flow exactly
    def play(self, parameters):

        #if bigblind not set, use default
        if self.bigblind == 0: self.bigblind = 10

        #we must have a regret manager and game abstractor
        regret_man = self.regret_man
        abstractor = regret_man.game_abstractor
        blind_game = True

        #how many players
        print("  How many players?: ", end="")
        nb_player = int(input())

        #start a new emulator
        max_round = 10
        initial_stack = 1000000
        sb_amount = self.bigblind / 2
        ante_amount = 0
        emulator = Emulator()
        emulator.set_game_rule(nb_player, max_round, sb_amount, ante_amount)

        #setup callidus
        callidus = EmulatorPlayer(self.regret_man)
        callidus.allow_retraining = False
        callidus.training_timelimit = 7
        callidus.verbose = True
        callidus.use_cfr = self.use_cfr
        callidus.use_monte = self.use_monte
        callidus.use_argmax = self.use_argmax

        #setup the players
        players = [callidus] + [InteractiveHuman(p == 0) for p in range(nb_player-1)]
        player_info = setup_emulator_players(emulator,players)

        #genreate an initial state for the game
        initial_state = emulator.generate_initial_game_state(player_info)

        #while we are still playing
        done = False
        while not done:

            ###get all info from the user right away##

            print("  What's my stack?: ",end="")
            callidus_stack = float(input())

            print("  Who is dealer?: ",end="")
            btn_pos = int(input())

            print("  Enter Hole: ", end="")
            hole_card = input().upper().split()

            ###now use that information###

            #we are actually setting the button one player behind
            #where we want it to start, because as soon as we start
            #the first round, it's going to shift forward
            if btn_pos == 0: btn_pos = nb_player - 1
            else: btn_pos -= 1

            #set blind positions appropriately
            sb_pos = btn_pos + 1
            bb_pos = sb_pos + 1
            if sb_pos >= nb_player: sb_pos, bb_pos = 0, 1
            if bb_pos >= nb_player: bb_pos = 0

            #update on table
            initial_state['table'].dealer_btn = btn_pos
            initial_state['table'].set_blind_pos(sb_pos,bb_pos)

            #start a new round
            game_state, events = emulator.start_new_round(initial_state)

            #ask for stack each time
            for p in game_state['table'].seats.players:
                if p.uuid == callidus.uuid: p.stack = callidus_stack
                else: p.stack = initial_stack

            #update all player stacks before game starts
            #here we can re-assign hole cards as needed
            player_info["p0"]["starting_hole"] = hole_card
            game_state = abstractor.assign_hole_cards(game_state, player_info)

            #focus training on this hole (getting a head start while we wait for other players)
            if callidus.allow_retraining: callidus.focus_training(game_state)

            #play each round until street is "FINISHED"
            community_card = []
            street = 0
            while game_state["street"] != Const.Street.FINISHED:

                #request next street
                street += 1
                game_state, events = emulator.run_until_round_finish(game_state, street)

                #get community cards
                print("  Add to Community [{}]: ".format([str(c) for c in community_card]), end="")
                new_community = input().upper().split()
                community_card += new_community

                #replace those cards on the table
                game_state = abstractor.assign_community_cards(game_state, community_card)

                #break if "quitting"
                if InteractiveHuman.QUIT_ALL: break

                #increase focus as we have moved the game state
                if callidus.allow_retraining: callidus.focus_training(game_state)

            #get game result
            InteractiveHuman.QUIT_ALL = False
            game_result = emulator._generate_game_result_event(game_state)
            print(str(game_result))

            #unfocus training until we know the next hole
            if callidus.allow_retraining: callidus.unfocus_training()

            #should we keep playing
            print("  Keep Playing?: ",end="")
            done = input().upper()[0] != "Y"

    #run a simulation using current regret manager
    def play2(self,parameters):

        #setup a new bot using our current regrets
        cfr_bot = EmulatorPlayer(self.regret_man)
        cfr_bot.allow_retraining = True #we can retrain as needed during simulations

        #if bigblind not set, use default
        if self.bigblind == 0: self.bigblind = 10

        #run some simulations in any case
        print("  Playing Against Callidus")

        stack_log = []
        game_result = None
        wins = 0
        games = 1
        rounds = 10
        winnings = 0
        
        #start progress bar right away so we know something is happening
        for round in range(games):
            config = setup_config(max_round=rounds, initial_stack=1000, small_blind_amount=self.bigblind / 2)
            config.register_player(name="p1", algorithm=cfr_bot)

            for i in range (2):
                config.register_player(name="p{}".format(i+2), algorithm=InteractiveHuman(i == 0))
            game_result = start_poker(config, verbose=0)

            print("GAME RESULTS:")
            for player in game_result['players']:
                print("Player {} Stack = {:4.2f}".format(player['uuid'],player['stack']))


    #show probs
    def show_probs(self,parameters):

        #generate new probs for range
        probs = self.regret_man.game_abstractor.build_hand_probs(int(parameters[0]))

        #display them in a readible format
        for ptype in probs:
            print("")
            print("{} Probs:".format(ptype))
            print("----------------------------")
            for hole in probs[ptype]:
                print("{}: {}".format(hole,str(probs[ptype][hole])))

    #show probs
    def monte(self,parameters):
        #get abstractor reference
        abstractor = self.regret_man.game_abstractor

        #generate new probs for range
        if parameters[0] == "BUILD":
            abstractor.monte_results = multi_monte(abstractor, 6,int(parameters[1]), abstractor.monte_results)
            print("")
        elif parameters[0] == "LOAD":

            #did user provide a specific monte file?
            montefile = 'monte.pkl'
            if len(parameters) > 1: montefile = parameters[1]

            #load that file
            print("Loading Monte File {} into Shared Memory Space...".format(montefile))
            with open(montefile,'rb') as fb:
                #load into a temporary array
                monte_results = pickle.load(fb)

                #create a shared memory space that matches size
                abstractor.monte_shm = shared_memory.SharedMemory(
                    create=True, 
                    size=int(np.product(multi_monte_shape()) * 4),
                    name = "SHARED_MONTE"
                )

                #create an np array on that shared memory space
                #then copy from shared to this one
                abstractor.monte_results = np.ndarray(multi_monte_shape(), dtype=np.int32, buffer=abstractor.monte_shm.buf)
                np.copyto(abstractor.monte_results, monte_results)
                del monte_results

        
        elif parameters[0] == "OPEN":

            #open our monte
            abstractor.open_monte("SHARED_MONTE")
            
        elif parameters[0] == "ADD":

            #user must supply a monte file
            montefile = 'monte.pkl'
            if len(parameters) > 1: montefile = parameters[1]

            #load that file into a separate reuslt
            print("Loading Monte File {}...".format(montefile))
            with open(montefile,'rb') as fb:
                monte_results = pickle.load(fb)

            #now combine those results with our current results
            for street in range(len(monte_results)):
                #for each hand in this street
                progressBar("Progress",street,4)
                for hand in range(len(monte_results[street])):
                    for comm in range(len(monte_results[street][hand])):
                        abstractor.monte_results[street][hand][comm][0] += monte_results[street][hand][comm][0]
                        abstractor.monte_results[street][hand][comm][1] += monte_results[street][hand][comm][1]


        elif parameters[0] == "SAVE":
            #did user provide a specific monte file?
            montefile = 'monte.pkl'
            if len(parameters) > 1: montefile = parameters[1]

            #save that file
            print("Saving Monte File {}...".format(montefile))
            with open(montefile,'wb') as fb:
                pickle.dump(abstractor.monte_results,fb)
        elif parameters[0] == "STATS":
            #print that out
            res = abstractor.monte_results
            stats = np.zeros(3,dtype = np.int64)
            sims = 0
            blanks = 1
            lows = 2
            for s in range(len(res)):
                for h in range(len(res[s])):
                    if res[s][h][-1][0] > 0:
                        stats[sims] += res[s][h][-1][0]
                        if res[s][h][-1][0] < 100:
                            stats[lows] += 1
                    else:
                        stats[blanks] += 1
            print("MONTE Stats:")
            print("Total Simulations: {}".format(stats[sims]))
            print("Low Count Simulations: {}".format(stats[lows]))
            print("Missing Simulations: {}".format(stats[blanks]))
        elif parameters[0] == "SHOW":

            #show only blanks?
            showblanks = False
            showhand = None
            if len(parameters) > 1: 
                if parameters[1] == "BLANK": showblanks = True
                else: showhand = parameters[1]

            #print that out
            res = abstractor.monte_results
            for s in range(len(res)):
                for h in range(len(res[s])):
                    if h % 10 < 9:
                        handname = abstractor.unpack_hand_profile(h)
                        if showhand == handname[0] or showhand == None:
                            if res[s][h][-1][0] > 0 and not showblanks:
                                print("street {}-{}-{}: {} | ALL = {} of {} = {:4.2f}".format(
                                    s,h,1600,
                                    handname,
                                    res[s][h][-1][1],
                                    res[s][h][-1][0],
                                    res[s][h][-1][1] / res[s][h][-1][0]
                                ))
                                for c in range(len(res[s][h])-1):
                                    if res[s][h][c][0] > 0:
                                        print("street {}-{}-{}: {} | {} = {} of {} = {:4.2f}".format(
                                            s,h,c,
                                            abstractor.unpack_hand_profile(h),
                                            abstractor.unpack_hand_profile(c),
                                            res[s][h][c][1],
                                            res[s][h][c][0],
                                            res[s][h][c][1] / res[s][h][c][0]
                                        ))
                            #there is some sparseness in the hand buckets - every 9th item is actually non-existent
                            elif res[s][h][-1][0] <= 0 and showblanks and h % 10 < 9:
                                print("street {}-{}-{}: {} | ALL = {} of {}".format(
                                    s,h,1600,
                                    abstractor.unpack_hand_profile(h),
                                    res[s][h][-1][1],
                                    res[s][h][-1][0]
                                ))
                                for c in range(len(res[s][h])-1):
                                    if res[s][h][c][0] > 0:
                                        print("street {}-{}-{}: {} | {} = {} of {}".format(
                                            s,h,c,
                                            abstractor.unpack_hand_profile(h),
                                            abstractor.unpack_hand_profile(c),
                                            res[s][h][c][1],
                                            res[s][h][c][0]
                                        ))

        elif parameters[0] == "COMM":
            #get community cards
            hole = gen_cards(parameters[1:3])
            comm = gen_cards(parameters[3:])

            #get monte for comm against all holes
            (wins,games,win_pct) = abstractor.get_montecomm(hole,comm)

            #print all that out
            print("Comm Monte {} of {} => {:4.2f}%".format(wins,games,win_pct * 100))

        elif parameters[0] == "TRANS":

            #translate a card profile
            profile = int(parameters[1])
            print(abstractor.unpack_hand_profile(profile))

        else:

            #get card from parameters
            hole = gen_cards(parameters[:2])
            comm = gen_cards(parameters[2:])

            #what is the street?
            street = 0
            if len(comm) == 3: street = 1
            elif len(comm) == 4: street = 2
            elif len(comm) == 5: street = 3

            #get monte for hole/comm
            (wins,games,win_pct) = abstractor.get_monte(hole,comm)

            #hand rank
            hand_rank = he.gen_hand_rank_info(hole,comm)
            print(hand_rank)

            #get hand profile
            hand_profile = abstractor.hand_profile(hole,comm)

            #now print everything about that profile (like the show command)
            print(abstractor.monte_results[street][hand_profile])


            #print all that out
            print("Monte {} of {} => {:4.2f}%".format(wins,games,win_pct * 100))


    #internally get or set a variable value
    def setvar(self, varname, newval = None):
        #these are translated as truths:
        truths = ["ON","TRUE","T","1"]
        returnval = None

        #are we setting the value of "M"
        if varname == "M":
            if newval != None: self.regret_man.relative_size = int(newval)
            returnval = self.regret_man.relative_size

        #system information - we can only return cannot set
        if varname == "CORES":
            #return the # of cores
            returnval = multiprocessing.cpu_count()

        #version information - we can only return cannot set
        if varname == "VERSION":
            returnval = "\n\n  browser: {}\n  python: {}\n  numpy: {}".format(self.browser.REGRETBROWSER_VERSION, sys.version,np.version.version)

        #use monte
        if varname == "USEMONTE":
            if newval != None:
                if newval in truths: self.use_monte = True
                else: self.use_monte = False
            returnval = self.use_monte

        #use cfr
        if varname == "USECFR":
            if newval != None:
                if newval in truths: self.use_cfr = True
                else: self.use_cfr = False
            returnval = self.use_cfr

        #bb value
        if varname == "BB":
            if newval != None:
                self.bigblind = float(newval)
            returnval = self.bigblind

        #shared identity
        if varname == "ID":
            if newval != None:
                self.identity = newval
            returnval = self.identity

        #nocatch value
        if varname == "NOCATCH":
            if newval != None:
                if newval in truths: self.catch_errors = True
                else: self.catch_errors = False
            returnval = self.catch_errors

        #argmax value
        if varname == "ARGMAX":
            if newval != None:
                if newval in truths: self.use_argmax = True
                else: self.use_argmax = False
            returnval = self.use_argmax

        #return the return value
        return returnval

    #set any variable value
    def set(self,parameters):

        #assume we don't set a value
        displayval = self.setvar(parameters[0],parameters[1])

        #if displayval is NONE then the value was not found
        if displayval == None:
            print("Unknown variable {}".format(parameters[0]))
        else:
            print("{} set to: {}".format(parameters[0],displayval))

    #get any variable value
    def get(self,parameters):

        #assume we don't set a value
        displayval = self.setvar(parameters[0],None)

        #if displayval is NONE then the value was not found
        if displayval == None:
            print("Unknown variable {}".format(parameters[0]))
        else:
            print("{} is currently set to: {}".format(parameters[0],displayval))
