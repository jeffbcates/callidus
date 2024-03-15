from browser_commands import BrowserCommands
from Regrets import RegretManager
from PokerAbstractor import PokerAbstractor
import numpy as np
import sys

#RegretBrwoser - functionality for interacting with regrets through the command line

class RegretBrowser:

    #constants
    REGRETBROWSER_VERSION = "1.0"

    #initialize browser
    def __init__(self):

        #setup some values
        self.regret_man = None

        #register our commands
        self.commands = {}

    #register a browser command
    def register_command(self,name,help,reference, params, details = []):

        #add this command to our command list
        self.commands[name.upper()] = {"help": help,"details":details,"reference": reference, "params": params}

    #welcome message
    def welcome(self):

        #print some basic information for the user
        print("Welcome to Regret Browser {}".format(self.REGRETBROWSER_VERSION))
        print("")
        print("Type any command at the prompt, or HELP for hints")
        print("")

    #prompt for command
    def prompt(self):
        #get display path for our path
        display_path = self.regret_man.game_abstractor.unpack_path_tuple([tuple([int(s) for s in sp.split(":")]) for sp in self.current_path])

        #display prompt and get input from user, which we'll return
        print("\\".join(["{} {}".format(display_path[px],self.current_path[px]) for px in range(len(self.current_path))]) + ">",end = "")
        user_input = input().upper().split()
        return user_input

    #regenerate node from path
    def gen_current_node(self, path):

        #try to generate the path, if there is an error return current path
        try:

            #create a real path from the string path
            #converting each A:B pair into a tuple
            real_path = [tuple(map(int,p.split(":"))) for p in path]

            #does the path exist?
            if self.regret_man.symm_tree.get(real_path) == None: real_path = None
        except:

            #there was an issue, set path to none
            real_path = None

        #return that path
        return real_path

    #pipe commands - unsplits and resplits commands by comma
    #to handle piping multiple commands
    def pipe(self,commands):
        #join by space, split by comma and then return the 2-d array of commands
        return [s.strip().split() for s in " ".join(commands).split(",")]

    #main entry for the browser
    def browse(self, regret_man = None, args = []):

        #print welcome
        self.welcome()

        #fix printing issues for floats
        np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

        #register commands
        self.browser_commands = BrowserCommands(self,regret_man)
        self.browser_commands.register()

        #setup starting path
        self.regret_man = regret_man
        self.current_node = []
        self.current_path = []

        #run commands until exit
        command = ""
        while command != "EXIT":

            #if initializing get command from args
            if len(args) > 0:
                user_input = self.pipe(args)
                args = []
            else: user_input = self.pipe(self.prompt())

            #step through all commands in user input
            for commands in user_input:
                #get command name
                command = commands[0]

                #if not found, let them know
                if command not in self.commands: 
                    print("Unknown command " + command)
                elif len(commands) < self.commands[command]["params"] + 1:
                    print("Incorrect number of parameters supplied to command, expected: " + str(self.commands[command]["params"]))
                else: 

                    #should we catch errors
                    if self.browser_commands.catch_errors:

                        #run the command - catch errors and display
                        try:
                            self.commands[command]["reference"](commands[1:])
                        except Exception as error:
                            print("ERROR RUNNING COMMAND: " + str(error))
                    else:

                        #run command without catching errors
                        self.commands[command]["reference"](commands[1:])

                    #regenreate node reference
                    self.current_node = self.gen_current_node(self.current_path)

#run our process
if __name__ == "__main__":
    
    #run browser with parameters
    rb = RegretBrowser()
    rb.browse(RegretManager(PokerAbstractor()),[s.upper() for s in sys.argv[1:]])

        