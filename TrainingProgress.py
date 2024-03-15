import time
from util import progressBar

class TrainingProgress():

    #track the # of regrets at the start of an epoch/training
    epoch_regrets = 0
    training_regrets = 0

    #track the # of iterations for each step within the epoch and each epoch, and total for training
    epoch_iterations = 0
    training_iterations = 0
    step_iterations = 0

    #track the start time of training and of eahc epoch
    epoch_start = 0
    training_start = 0
    step_start = 0
    time_limit = 0

    #how many simulations did we run (step)
    simulations = 0

    #for training - how many epochs and what are their sizes
    epochs = 0
    epoch_size = 0

    #for multi-training scenarios, we write to an array
    external_status_array = None
    status_array_index = 0

    #initialize our training reporting object
    def __init__( self, external_status_array = None, status_array_index = 0):

        #when we are multi training, we actually don't write anything out to the console
        #we just update the external status array with our status
        self.external_status_array = external_status_array
        self.status_array_index = status_array_index

    #start training
    def start_training(self, regret_man, epochs, epoch_size, time_limit):

        #save off # of epochs and epoch size
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.simulations = 0
        self.time_limit = time_limit

        #lookup training regrets
        self.training_regrets = regret_man.eval_training()[2]
        self.epoch_regrets = self.training_regrets

        #record start times
        self.training_start = int(time.time())
        self.epoch_start = self.training_start
        self.step_start = self.training_start
        
    #at the end of an epoch - report on the results of that epoch
    #and return the # of new regrets calculated
    def report_epoch_results(self, regret_man, epoch, verbosity=2):

        #determine new regrets that were added during this training epoch
        new_regrets = regret_man.eval_training(False)[2]

        #calculate epoch speed
        epoch_speed = time.time() - self.epoch_start
        if epoch_speed <= 0: epoch_speed = 0.01
        epoch_speed = ( new_regrets - self.epoch_regrets ) / epoch_speed

        #if mutli-training write out stats to array
        if self.external_status_array != None:
            self.external_status_array[self.status_array_index] = epoch #current epoch
            self.external_status_array[self.status_array_index + 1] = self.epoch_size #current step
            self.external_status_array[self.status_array_index + 2] = self.epoch_size #total steps
            self.external_status_array[self.status_array_index + 3] = self.epoch_regrets #total regrets for the epoch
            self.external_status_array[self.status_array_index + 4] = time.time() - self.epoch_start #time elapsed for epoch

        #print some stats on the epoch
        if verbosity == 2:
            progressBar("Epoch {} of {}".format(epoch,self.epochs),self.epoch_size,self.epoch_size,
                "{} Regrets @ {:4.2f} Regrets/Sec".format(
                ( new_regrets - self.epoch_regrets ),
                epoch_speed
            ))
            print("")

        #the epoch is done, so record new regrets as current regrets
        #and current time as start time
        self.epoch_regrets = new_regrets
        self.epoch_start = int(time.time())

        #reset # of epoch iterations
        self.epoch_iterations = 0

        #below we figure out if training has completed
        #we print out some information to the user (if needed)
        #and return that completion flag to the caller
        done_training = False
        if self.time_limit > 0:
            if time.time() - self.training_start > self.time_limit:
                if verbosity == 1: print("")
                if verbosity > 0: print("Ending Training - Time Limit Reached")
                done_training = True
        elif epoch >= self.epochs and self.epochs > 0:

            #end training if we have reached the # of epochs
            if verbosity == 1: print("")
            if verbosity > 0: print("Completed {} Epochs".format(self.epochs))
            done_training = True

        #reurn our new regrets for later use
        return (new_regrets, done_training)

    #report on training results
    def report_training_results(self, regret_man, verbosity):

        #final time taken
        timetaken = int(time.time()) - self.training_start

        #how many regrets?
        training_regrets = self.epoch_regrets - self.training_regrets

        #print out results and enter regret browser
        if verbosity > 0: 
            print("{} simulations in {:4.2f} sec = {:4.2f} hand/sec".format(self.simulations, timetaken,self.simulations/timetaken))
        if verbosity > 0: 
            print("{} regrets in {:4.2f} sec = {:4.2f} regret/sec".format(training_regrets, timetaken,training_regrets/timetaken))

        #return in case caller needs this information as well
        return (self.simulations, training_regrets, timetaken)

    #within each epoch we need to report on the progress of the epoch
    #we do that with this function (duh lol)
    def report_epoch_progress(self, regret_man, epoch, step, path=None, verbosity = 2, focused = 0, iterations = 0):

        #one more simulation
        self.simulations += 1

        #add to iterations
        self.epoch_iterations += iterations
        self.training_iterations += iterations
        self.step_iterations = iterations

        #how long did this step take
        epoch_timetaken = (time.time() - self.epoch_start)
        timetaken = (time.time() - self.step_start)
        self.step_start = time.time()

        #a simple message for more info
        more_info = ""
        if focused: more_info = "Focus Level {}".format(focused)
        else: more_info = "{:4.1f} secs, iters: {} @ {:4.2f}/sec".format(timetaken, self.epoch_iterations,self.epoch_iterations / epoch_timetaken)

        #if mutli-training write out stats to array
        if self.external_status_array != None:
            self.external_status_array[self.status_array_index] = epoch #current epoch
            self.external_status_array[self.status_array_index + 1] = step #current step
            self.external_status_array[self.status_array_index + 2] = self.epoch_size #total steps
            self.external_status_array[self.status_array_index + 3] = self.epoch_iterations #total regrets for the epoch
            self.external_status_array[self.status_array_index + 4] = time.time() - self.epoch_start

        #as a test - don't show progress
        if verbosity == 2: progressBar("Epoch {} of {}".format(epoch,self.epochs),step,self.epoch_size, more_info)

        #if verbosity is 1 - show only a single line for all training
        elif verbosity == 1:

            #if a path was provided, we should show that in more info
            if path != None:

                #we either have an initial state, or we have a starting hole
                #either way - let's show the information set from right after hole cards
                #were provided to the trainee
                infoset = regret_man.get_information_set(None,None,False, regret_path = path)
                more_info = str(infoset.get_average_strategy()) 
                    
            #if epochs is not zero, calc progress by epochs
            if self.epochs > 0:
                progress_value = (epoch - 1) * self.epoch_size + step
                progress_total = self.epochs * self.epoch_size
            else:
                progress_value = time.time() - self.training_start
                progress_total = self.time_limit
                        
            #display that info
            progressBar("Training",progress_value,progress_total,more_info,barLength = 10)