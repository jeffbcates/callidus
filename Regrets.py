from typing import List, Dict
from symmetric_tree import SymmetricTree
import random
import numpy as np
import sys
import pickle
import json
import time
import glob
import os

#this is required for shared memory accessed between processes
from multiprocessing import shared_memory as mem

#these are required for shared read/write to files in windows
#import msvcrt

REGRET_DEBUG = False
class InformationSet():

    #initialize a new infoset
    def __init__(self, path:list, symm:SymmetricTree, create:bool = True):

        #save a reference to the symm tree and path
        self.path = path
        self.symm = symm
        self.create = create

        #get shape array reference
        shape = symm.shape()

        #if the last item in our path is an action (not a hand profile)
        #then let's add a fake hand profile to store the infoset 
        #this keeping the shape symmetric but allowing non-leaf infosets
        #if path[-1][1] == 1: path += [(shape[2]-1,2)]

        #the # of actions is stored at the leaf level of infosets
        self.num_actions = shape[RegretManager.REGRET_PATH * SymmetricTree.SHAPE_SIZE]

        #only create if we are creating

        #if this is a new infoset, set everything up for it
        if self.create:
            if self.symm.get(self.path + RegretManager.PATH_STAT_LOC_READS, default=0, set_default = True) == 0:

                #write default regrets and strategy (using get / default options -> allows us to write 
                self.symm.set( self.path + RegretManager.PATH_INFOSET_REGRETS, np.zeros(self.num_actions) )
                self.symm.set( self.path + RegretManager.PATH_INFOSET_STRAT, self.get_default_strategy() )

                #we have now read once and written once
                self.symm.set(self.path + RegretManager.PATH_INFOSET_STAT,[1,1])

    #return our reads
    def reads(self):
        return self.symm.get( self.path + RegretManager.PATH_STAT_LOC_READS,0 )

    #return our writes
    def writes(self):
        return self.symm.get( self.path + RegretManager.PATH_STAT_LOC_WRITES,0 )

    #return our cumulative regrets
    def regrets(self):
        #get regrets
        regrets = self.symm.get( self.path + RegretManager.PATH_INFOSET_REGRETS,items=self.num_actions)

        #if not initialized, set to zeros -> this will happen when reading from a path that doesn't exist
        #as a "reader" not a trainer
        if type(regrets) == type(None): regrets = np.zeros(self.num_actions)

        #return those raw regrets
        return regrets

    #return our strategy
    def strategy(self):
        #get our strategy
        strat = self.symm.get( self.path + RegretManager.PATH_INFOSET_STRAT,items=self.num_actions)

        #if not initialized, set to default -> this will happen when reading a path that hasn't been
        #created yet (not trained yet)
        if type(strat) == type(None):  strat = self.get_default_strategy()

        #return a copy of our strategy
        #so that any manipulations by caller do not change the original
        return np.copy(strat)

    #return our actions
    def actions(self):
        return self.num_actions

    #normalize a strategy
    def normalize(self, strategy: np.array) -> np.array:
        #normalize a strategy. If there are no positive regrets,
        #use a uniform random strategy
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = self.get_default_strategy()
        return strategy

    #return the default strategy
    def get_default_strategy(self):
        return np.array([1.0 / self.num_actions] * self.num_actions)

    def get_strategy(self, reach_probability: float, depth: int) -> np.array:
        #Return regret-matching strategy
        #JBC 10/10/20 -> normalize regrets, but continue to use default
        #until a clear best path emerges (ie sum of regrets is positive)
        #BE SURE TO MAKE A COPY OF REGRETS SO WE DON'T OVERWRITE THEM!!!
        strategy = np.maximum(0, np.copy(self.regrets()))
        strategy = self.normalize(strategy)

        #do not update strategy if the reach probability is ZERO
        #because the result is not going to change - this can reduce space in the model
        #for those paths we NEVER EVER reach, and can also speed up training by not writing
        #values unnecessarily
        if reach_probability > 0:

            #update strategy sum with new strategy
            self.symm.set( self.path + RegretManager.PATH_INFOSET_STRAT, reach_probability * strategy, SymmetricTree.MATH_ADD)

            #there is one more read
            self.symm.set( self.path + RegretManager.PATH_STAT_LOC_READS,1,SymmetricTree.MATH_ADD)

        #now that we have updated strategy, return normalized strategy sum (i.e. average strategy)
        return self.get_average_strategy()

    def update_regrets(self, counterfactual_values: np.array):

        #there is one more write
        self.symm.set( self.path + RegretManager.PATH_STAT_LOC_WRITES,1,SymmetricTree.MATH_ADD)

        #update regret if we were able to call this value
        #JBC: 9/11/20 - added this to prevent negative regret for actions not taken, does this work?
        #i am wondering if i should actually use the reach probability of the individual regret (strategy) instead
        #if counterfactual_values[i] != 0:

        #update regrets of info set to our regret
        self.symm.set( 
            self.path + RegretManager.PATH_INFOSET_REGRETS,
            counterfactual_values,
            SymmetricTree.MATH_ADD
        )

    #return the average strategy (no updating here)
    #if for some reason the strategy is not found we return default
    def get_average_strategy(self) -> np.array:

        #return our strategy, but normalized
        return self.normalize(self.strategy().copy())

    #return current regrets
    def get_regrets(self) -> np.array:

        #return our regrets, but normalized
        #BE SURE TO MAKE A COPY SO WE DON'T OVERWRITE CUMULATIVE REGRETS
        return self.normalize(self.regrets().copy())


#strategy manager abstracts the logic for tracking strategy at each iteration of CFR
class StrategyManager():

    #initialize the strategy manager for this strategy
    def __init__(self, strategy: np.array,action_sets: int):
        super().__init__()
        self.strategy = strategy
        self.counterfactual_values = np.zeros(action_sets)

    #record a counterfactual value for a given action (index)
    def set_counterfactual_value(self,action: int, value: float):
        self.counterfactual_values[action] = value

    #calculate regret given reget values and strategies
    def get_regret(self):
        #dot multiplies each counterfactual value by its corresponding strategy
        #then sums those values to get a total
        return self.counterfactual_values.dot(self.strategy)

    #update info set regrets given reach probability
    def update_regrets(self, info_set: InformationSet, reach_probability: float):
        #JBC 10/6/20 -> moved this reach_probability check from infset update_regrets to here
        #because we are no longer passing reach_probability to infoset, and instead are apply the calc here

        #if our reach probability is ZERO, then we don't need to write regrets
        #because the result is not going to change - this can reduce space in the model
        #for those paths we NEVER EVER reach, and can also speed up training by not writing
        #values unnecessarily
        if reach_probability == 0: return

        #get our regret
        current_regret = self.get_regret()

        #JBC 10/6/20 -> calculate counterfactual values based on reach probability and current strategy
        #which will prevent paths we did not take from attributing value (negative or positive)
        #adjusted_regrets = self.strategy * reach_probability * (self.counterfactual_values - current_regret)

        #JBC 10/9/20 -> back to old way for testing
        #note the additional np where function prevents us from adjusting regrets that were not actually triggered
        #adjusted_regrets = reach_probability * (self.counterfactual_values - current_regret) * np.where(self.strategy > 0, 1, self.strategy)

        #JBC 10/9/20 -> for below we are not normalizing to 1 - this will make the adjustments to regret smaller
        #adjusted_regrets = reach_probability * (self.counterfactual_values - current_regret)

        #JBC 11/9/20 -> do not adjust strategies that were not used (strategy is ZERO)
        #this concept works fine as long as we are not normalizing remaining startegies 
        #which tilts our training too quickly to the paths that are available to take
        adjusted_regrets = reach_probability * (self.counterfactual_values - current_regret) * np.where(self.strategy > 0, 1, self.strategy)

        #update infoset with regret
        info_set.update_regrets(adjusted_regrets)

#regret manager abstracts the logic for calculating and tracking counterfactual regrets, and strategy(probabilities)
#so we can test between different games without needing to abstract our CFR agent
class RegretManager():

    #the types of our paths
    HOLE_PATH = 0
    LEAF_PATH = 1
    POS_PATH = 2
    ACTION_PATH = 3
    HAND_PATH = 4
    STRENGTH_PATH = 5
    SUITCOUNT_PATH = 6
    INFOSET_PATH = 7
    REGRET_PATH = 8
    STRAT_PATH = 9
    STAT_PATH = 10

    #specific definition locations
    REGRET_LOC = 0
    STRAT_LOC = 1
    STAT_LOC = 2

    #definitions of statistic values
    STAT_LOC_READS = 0
    STAT_LOC_WRITES = 1

    #predefine some common paths for statistics
    PATH_STAT_LOC_READS = [(STAT_LOC, INFOSET_PATH), (STAT_LOC_READS, STAT_PATH)]
    PATH_STAT_LOC_WRITES = [(STAT_LOC, INFOSET_PATH), (STAT_LOC_WRITES, STAT_PATH)]

    #predefine some common paths for regrets and strategies
    PATH_INFOSET_REGRETS = [(REGRET_LOC, INFOSET_PATH), (0, REGRET_PATH)]
    PATH_INFOSET_STRAT = [(STRAT_LOC, INFOSET_PATH), (0, STRAT_PATH)]
    PATH_INFOSET_STAT = [(STAT_LOC, INFOSET_PATH), (0, STAT_PATH)]

    #what is our relative size (the size of "m")
    relative_size = 1000

    #initialize symm tree shape from M constant
    def setup_symmtree_shape(self):
        #our symmetric tree shape
        m = self.relative_size
        symmtree_shape = [
            (2704,0, 1) , #0: hole cards - 52*52 + 1 (for infoset)                                                   -> 1 path
            (4,0, 600*m), #1: split between leaf and non-leaf paths (0 = leaf, 1 = vilan, 2=hero, 3 = end of street)  -> 300 million paths
            (3,0, 2704 * 3), #2: player position -> EARLY, MIDDLE, LATE                                              -> 3 for each root
            (14,0, 300*m), #3: action histories - 14 - O, C, R0 - R9                                                 -> 150 million paths
            (10,0, 70*m), #4: hand profile -> the hand of the card                                                    -> 30 million paths
            (10,0, 70*m), #5: strength profile -> the strength of the card                                            -> 30 million paths
            (16,0, 70*m), #6: suit profile -> determine flush / profile                                               -> 30 million paths
            (3,0, 250*m), #infoset arrays - cumulative regrets, strategy sum, and statistics                         -> 150 million paths
            (7,1, 250*m), #array of regrets                                                                            -> 30 million paths
            (7,1, 250*m), #array of strategy                                                                           -> 30 million paths
            (2,0, 250*m) #read and write counts are INT                                                               -> 150 million paths
            
        ]

        #save the shape for later use (we won't allocate memory until a load or attach
        #method is called
        self.symmtree_shape = symmtree_shape

    #initialize the regret manager
    def __init__(self, game_abstractor, file=None):
        super().__init__()
        self.infoset_map = {}

        #create a new empty symmetric tree
        
        #track the last time we loaded each regret path file
        #so we can decide if it needs reloading before reloading it
        self.regret_decay = {}

        #for testing
        self.regret_log = []
        
        #save the game abstractor (this converts game states into a regret path for us)
        self.game_abstractor = game_abstractor

        #previously we allocated or attached to symmetry tree right here
        #now we are going to just set its value to None
        self.symm_tree = None

        #for evaluation
        self.added_regrets = 0

        #for testing, we track training iterations on the regret manager
        self.training_iteration = 1

        #initialize shared identity logic for regret locks
        self.shared_identity = None
        self.regret_locks = []


    #get an information set given the game state
    def get_information_set(self, game_state, player, save_set = True, regret_path = None) -> InformationSet:
        #get regret path
        #for testing - add training iteratoin to regret path to keep each regret unique
        if regret_path == None:
            regret_path = self.game_abstractor.gen_regret_path(game_state, player)
        
        #for testing
        #self.regret_log.append(regret_path)

        #return an infoset from this path
        #the infoset will allocate space in the symmetric tree as needed
        return InformationSet(path=regret_path, symm=self.symm_tree, create=save_set)


    #return regrets dictionary
    def get_regrets(self) -> Dict:
        return self.infoset_map

    #get the training of a regret node
    def eval_training(self, quick=False):
        #the total number of leaf nodes is actually stored as the last value of the array (where we track size / shape)
        (_,_,_,regret_nodes) = self.symm_tree.levelinfo(RegretManager.REGRET_PATH)
        (strategy_size,_,_,strategy_nodes) = self.symm_tree.levelinfo(RegretManager.STRAT_PATH)
        trained_nodes = 0

        #if we have time, calculate trained nodes:
        if not quick:
            #reshape the array as 2d
            view = self.symm_tree._arrays[RegretManager.STRAT_PATH][0:strategy_nodes * strategy_size].reshape(-1,strategy_size)

            #trained nodes are those where at least 1 action has pulled away from average strategy
            #if all strategies are closly aligned, the node is really not trained yet
            trained_nodes = np.count_nonzero(np.max(view,axis=1) - np.min(view,axis=1) > 0.01)

        #this needs to be reevaluated
        #for now just return the # of defined information set arrays div 3  (since there are 3 for each infoset)
        return regret_nodes, strategy_nodes, trained_nodes

    #create regrets on disk
    def create(self, filename = None):

        #create a new symmetric tree on disk if file proided
        if filename != None: 
            self.setup_symmtree_shape()
            self.symm_tree = SymmetricTree(shape=self.symmtree_shape,filename=filename, ondisk=True)
            self.on_disk = True
            self.shared_memory = False
            self.shared_space = filename
        else: 
            self.setup_symmtree_shape()
            self.symm_tree = SymmetricTree(shape=self.symmtree_shape,namespace="regrets_tree")
            self.on_disk = False
            self.shared_memory = True
            self.shared_space = "regrets_tree"



    #open regrets on disk
    def open(self, filename):

        #create a new symmetric tree on disk
        self.symm_tree = SymmetricTree(filename=filename, ondisk=True)
        self.shared_space = filename

        #we are on disk
        self.on_disk = True
        self.shared_memory = False

    #save regrets to a file
    def save(self, verbose = True):

        #save our symmetric tree
        self.symm_tree.save("regrets", verbose)

    #attach regrets to memory
    def attach(self, verbose = True):

        #create a new symmetric tree
        self.symm_tree = SymmetricTree(namespace="regrets_tree")

        #we are lodaed to shared memory
        self.shared_memory = True
        self.on_disk = False
        self.shared_space = "regrets_tree"
        
    #load all regrets at once
    def load(self, verbose = True):

        #create a new symmetric tree
        self.setup_symmtree_shape()
        self.symm_tree = SymmetricTree(shape=self.symmtree_shape,namespace="regrets_tree")

        #we are lodaed to shared memory
        self.shared_memory = True
        self.on_disk = False
        self.shared_space = "regrets_tree"
        
        #load our symmetric tree
        self.symm_tree.load("regrets",verbose = True)

    #interal - physical file path from regret path
    def regret_file_path(self, path):
        #create a joined path and return it
        return "regrets\\" + "\\".join([str(p) for p in path])

    #define our shared identity (will also load from shared memroy - must have already been initialized)
    def set_shared_identity(self, shared_identity):
        self.shared_identity = shared_identity
        self.regret_locks = mem.ShareableList(name="REGRET_LOCKS")

    #internal - lock or unlock a path
    def lock(self, path):

        #no need to lock if not sharing identity
        if self.shared_identity == None: return True

        #no need to check if anyone else is using this path
        #just mark that we are using it and return true
        self.regret_locks[self.shared_identity] = str(path[0])
        return True

    #internal - lock or unlock a path
    def unlock(self, path):

        #no need to unlock if not sharing identity
        if self.shared_identity == None: return True

        #just clear our current path
        self.regret_locks[self.shared_identity] = ""

        #always success
        return True

#write to a file
def _write_shared_file(filename, contents):

    #assume the file exists
    file_access = win32file.OPEN_EXISTING
    if not os.path.exists(filename): file_access = win32file.CREATE_NEW

    #create / open file
    handle = win32file.CreateFile(
        filename,
        win32file.GENERIC_WRITE,
        win32file.FILE_SHARE_DELETE | win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
        None,
        file_access,
        0,
        None
    )

    #detach handle
    detached_handle = handle.Detach()

    #get a file descriptor
    descriptor = msvcrt.open_osfhandle(detached_handle, os.O_BINARY)

    #open the file descriptor
    with open(descriptor,'wb') as fp:
        pickle.dump(contents,fp)
    

#read from a file
def _read_shared_file(filename):

    #if the file does not exist, return None
    if not os.path.exists(filename): return None

    #create / open file
    handle = win32file.CreateFile(
        filename,
        win32file.GENERIC_READ,
        win32file.FILE_SHARE_DELETE | win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
        None,
        win32file.OPEN_EXISTING,
        0,
        None
    )

    #detach handle
    detached_handle = handle.Detach()

    #get a file descriptor
    descriptor = msvcrt.open_osfhandle(detached_handle, os.O_BINARY)

    #open the file descriptor
    with open(descriptor,'rb') as fp:
        results = pickle.load(fp)
    
    #return that value
    return results