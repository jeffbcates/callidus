import time
import multiprocessing
from multiprocessing import Process
from multiprocessing import shared_memory

class MultiTest:

    TestName = ""

    def fun(self,process_name):
        #setup
        shared_status = shared_memory.ShareableList(name="PROCESS_TEST")

        #do the work
        for x in range(1000):
            shared_status[int(self.TestName)] = x
            for y in range(1000):
                for z in range(1000):
                    b = 1
                    a = b
                    b = a
    


class TestMain():

    def startfun(c):
        mt = MultiTest()
        mt.TestName = str(c)
        mt.fun(c)

    def inmain(self):

        #get our number of cores
        cores = multiprocessing.cpu_count()

        #start shared memory
        try:
            shared_mem = shared_memory.ShareableList([0] * cores, name="PROCESS_TEST")
        except:
            shared_mem = shared_memory.ShareableList(name="PROCESS_TEST")



        #all our processes
        processes = []

        #get number of cores
        print("Attaching to {} Cores!".format(cores))

        #start each process
        for c in range(cores):
            mt = MultiTest()
            mt.TestName = str(c)
            p = Process(target=TestMain.startfun, args=("{}".format(c),))
            p.start()
            processes.append(p)

        #wait until both processes complete, reporting on their status
        any_alive = True

        while any_alive:
            #get sum of all shared memory values
            current_count = sum(shared_mem)
            total_count = cores * 1000

            #check on status
            print("Status: {:4.2f}% : {}".format(current_count / total_count,shared_mem), end="\r")

            #wait a second
            time.sleep(1)

            #are any alive
            any_alive = sum([1 for b in processes if b.is_alive() == True])

        print("")
        print("All Done!")

def main():
    t = TestMain()
    t.inmain()

if __name__ == '__main__':
    main()