import numpy as np
from multiprocessing import shared_memory as mem
import pickle
from util import progressBar
class SymmetricTree:

    #our math operations
    MATH_ADD = 0
    MATH_SUB = 1
    MATH_MULT = 2
    MATH_DIV = 3
    MATH_INTDIV = 4
    MATH_EXP = 5

    #shape size
    SHAPE_SIZE = 4

    #the location of items stored in shape array
    LOC_SHAPE = 0
    LOC_TYPE = 1
    LOC_TOTAL = 2
    LOC_USED = 3

    #shape can only be 1000 items long
    MAX_SHAPE_ARRAY_SIZE = 1000

    NP_DATUM_SIZE = 4 #32 bit unsigned integer

    #max individual array size is limited only by size of NP_DATUM_SIZE
    #MAX_INDIVIDUAL_ARRAY_SIZE = 256**4 - 2 #1024 * 1024 * 1000 // 4 - 2
    MAX_INDIVIDUAL_ARRAY_SIZE = 1024 * 1024 * 1000 - 2

    #initialize the tree
    def __init__( self, shape = None, namespace = None, filename = None, ondisk = False):

        #references to our internal numpy arrays
        self._arrays = []
        self._types = []
        self._managers = []
        self._type_managers = []
        self._shape = None
        self._shape_shm = None
        self.creator = False
        self.namespace = None
        self.ondisk = ondisk

        #load from file if a file is provided
        #obviously if we are loading from a file
        #we are not creating new, and we are not attaching to a namespace
        if filename != None and ondisk == False:
            self.load(filename,namespace)

        #if no shape is specified but a namespace is specified
        #then we should load from the namespace
        elif shape == None and namespace != None: 
            self._load_from_namespace(namespace)

        #if only a filename and onsidsk is set, we are openeing
        elif shape == None and ondisk == True:

            #open on disk
            self.open(filename)
        
        #if only a shape is specified, we are creating from scratch
        elif shape != None:

            #create ondisk
            if ondisk: self.create(shape, namespace, filename)
            else: self.create(shape,namespace)

        #if we got this far we have created an empty symmetric tree
        #that will later need to be loaded from a namespace or created


    #return our shape
    def shape(self):
        return self._shape

    #unlink - release global shared memory
    def unload(self):

        #unlink and close the shape array
        #if it is defined
        if self._shape_shm != None:
            if self.creator: self._shape_shm.unlink()
            self._shape_shm.close()

        #run through all managers and close and pop them
        while len(self._managers) > 0:
            shm = self._managers.pop()
            if self.creator: shm.unlink()
            shm.close()

        #we no longer have managers, a namespace etc
        self.namespace = None
        self.creator = False

    #get level information from shape array
    def levelinfo(self, level):
        """levelinfo Function

        Returns a tuple describing the shape and size of the level within the symmetric tree.

        Returns:
            tuple {integer} -- (shape, type, total, used)

        """

        return  (
                    self._shape[level * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_SHAPE], #shape
                    self._shape[level * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TYPE], #type
                    self._shape[level * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL], #total
                    self._shape[level * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_USED] #used
                )

    #return size of individual level
    def levelsize(self, level):

        #get level info and calculate available from total - used
        (shape,_,total,used) = self.levelinfo(level)
        used = int(self._shape[level * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_USED])
        total = total / shape
        available = total - used
        return (total,used,available)

    #return fill of individual level
    def leveldensity(self, level):

        #get the array size and shape
        (shape,_,_,size) = self.levelinfo(level)

        #we return the total sparsity plus the sparsity of each shape type
        results = []
        for c in range(shape):
            results.append(np.count_nonzero(self._arrays[level][c:size*shape:shape]))

        #add one additional result for total sparsity
        results.append(sum(results))

        #return that array
        return results

    #return detailed information about the sizes of the tree
    def size(self):

        #our return value is a dictionary
        info = []

        #step through each array
        for c in range(len(self._arrays)):

            #get level info
            (shape,dtype,total,used) = self.levelinfo(c)
            available = total // shape - used

            #describe this array
            arr = {

                "shape":shape,
                "type":dtype,
                "total":total // shape,
                "used":used,
                "available": available
            }

            #add to return value
            info.append(arr)

        #return that information
        return info

    #save to file
    def save(self, filename, verbose=False):

        #start a progress bar
        progressBar("Saving " + filename,0,len(self._arrays)+1)

        #save shape array
        np.save(filename + ".shape",self._shape,False,False)

        #create a new set of smaller arrays
        #but referencing the memory blocks of the full sized array in memory
        for ax in range(len(self._arrays)):
            #read level info
            (shape,dtype,total,entries) = self.levelinfo(ax)

            #get array, size and length
            a = self._arrays[ax]
            length = shape * entries
            size = length * SymmetricTree.NP_DATUM_SIZE

            #if we are using shared memory we can re-use the memory block
            #if we are NOT using shared memory, we have to create a new array
            if self.namespace != None:
                #create a new smaller array using the same underlying memory buffer as original array
                new_a = np.ndarray((length,), dtype=a.dtype, buffer=self._managers[ax].buf)

            else:
                #create a numpy array view pointed back to original array for only the used portion of the array
                new_a = a[:length]

            #progress bar
            progressBar("Saving " + filename,2 + ax,len(self._arrays)+1)

            #do the actual saving to file -> there are two arrays to save (data and type)
            np.save("{}.{}".format(filename,ax),new_a,False,False)

            #now do all the same steps for the type array if this array can hold indexes
            if dtype == 0 and self._type_managers[ax] != None:

                #get our type array reference
                t = self._types[ax]

                #get a smaller array view of only where we've stored data
                if self.namespace != None: new_t = np.ndarray((length,), dtype=t.dtype, buffer=self._type_managers[ax].buf)
                else: new_t = t[:length]

                #save the type array to file
                np.save("{}.{}t".format(filename,ax),new_t,False,False)

    #load from a file into a namespace
    #or from a namespace into us locally 
    def load(self, filename = None, namespace = None, verbose = False):

        #are we loading from a file (otionally into a namespace) or from a namespace
        if filename != None: self._load_from_file(filename,namespace, verbose)
        elif namespace != None: self._load_from_namespace(namespace)

    #open a file for RW operations - this is using
    #a memory mapped numpy array - bascically the same as shared virtual memory
    def open(self, filename, verbose=False):

        #open the memory-mapped shape array
        self._shape = np.load('{}.shape.npy'.format(filename),mmap_mode ='r+',allow_pickle = False,fix_imports = False)

        #step through all sizess in the shape
        #and open their corresponding array on disk
        for sx in range(self._shape[-1]):

            #get level info
            (shape,dtype,length,used) = self.levelinfo(sx)

            #open the array
            self._arrays.append(np.load('{}.{}.npy'.format(filename,sx),mmap_mode ='r+',allow_pickle = False,fix_imports = False))

            #for integer type levels, open the type array
            #and for floats, add a blank type array (there is no path type stored at the float level)
            if dtype == 0:
                self._types.append(np.load('{}.{}t.npy'.format(filename,sx),mmap_mode ='r+',allow_pickle = False,fix_imports = False))
            else:
                self._types.append([])



    #load from a file (optionally into a namespace)
    #load from file -> sadly, right now this will take 2x memory because
    #we don't have a way to load from file into a pre-allocated array
    def _load_from_file(self, filename, namespace = None, verbose=False):


        #load the shape array into a temporary array
        #and use that to create from definition
        shape = np.load(filename + '.shape.npy',allow_pickle = False, fix_imports = False)

        #progress
        progressBar("Loading " + filename,1,1 + shape[-1])

        #if the shape file has a different # of levels than our shape, we need to recreate our shape
        if shape[-1] != self._shape[-1]:
            print("Recreating symmetric tree to hold new shape")
            self.create(shape, namespace)

        #now, copy the shape to our new shape
        self._shape[:] = shape[:]

        #track if any levels are expanded / contracted, and write that out at the end
        level_adjustments = []

        #now for all our arrays (which are created now that we ran create_from_def):
        for ax in range(len(self._arrays)):

            #load this array using memory map in case its huge
            a = np.load("{}.{}.npy".format(filename,ax),mmap_mode ='r',allow_pickle = False,fix_imports = False)

            #copy into appropriate array from memory mapped file
            self._arrays[ax][:len(a)] = a[:]

            #compare our array with the declared length of the loaded array
            if len(self._arrays[ax]) > self._shape[ax * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL]:
                level_adjustments.append("Level {} expanded.  Adjusting shape".format(ax))
                self._shape[ax * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL] = len(self._arrays[ax])
            elif len(self._arrays[ax]) < self._shape[ax * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL]:
                level_adjustments.append("Level {} contracted.  Adjusting shape".format(ax))
                self._shape[ax * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL] = len(self._arrays[ax])

            #now immediately free up anything that was allocated for the memory mapped array
            del a

            #if integer shape, load a type array (for path types - since we store paths in our int arrays)
            if self._shape[ax * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TYPE] == 0:
                #load this array using memory map in case its huge
                #then copy its values into our corresponding shared type array
                #and just like the regular array immediately delete the memory mapped array freeing space
                t = np.load("{}.{}t.npy".format(filename,ax),mmap_mode ='r',allow_pickle = False,fix_imports = False)
                self._types[ax][:len(t)] = t[:]
                del t

            #progress bar for user
            progressBar("Loading " + filename,2 + ax,1 + shape[-1])

        #now, write out any adjustments made as needed
        print("")
        for adjustment in level_adjustments:
            print(adjustment)

    #create a new tree from a shape
    #JBC: 2020-11-05 - reviewed for packed index use
    def _load_from_namespace(self, namespace):

        #if we currently have a namesapce
        #release it
        if self.namespace != None: self.unload()

        #we now have a namespace
        #but we are not the creator
        self.namespace = namespace
        self.creator = False

        #clear current data structure
        _arrays = []

        #get our shape from shared list
        self._shape_shm = mem.SharedMemory(name = "{}_s".format(namespace))
        self._shape = np.ndarray((SymmetricTree.MAX_SHAPE_ARRAY_SIZE,),dtype=np.uint32,buffer = self._shape_shm.buf)
        shape = self._shape

        #step through all sizess in the shape
        #and create an array of max size to hold that shape
        #plus a few additional values to help manage the array
        for sx in range(shape[-1]):

            #get array info
            (shape,dtype,length,used) = self.levelinfo(sx)

            #attached to shared memory within namespace
            shm = mem.SharedMemory(name = "{}_{}".format(namespace,sx))

            #save reference to the memory manager
            self._managers.append(shm)

            #create the array using shared memory
            if dtype == 0:
                #get the path array as unsigned int
                path_array = np.ndarray((length,), dtype=np.uint32, buffer=shm.buf)

                #now we need to attach to the type array
                #which allows us to unpack the paths properly
                tshm = mem.SharedMemory(name = "{}_{}t".format(namespace,sx))
                type_array = np.ndarray((length,),dtype=np.uint8,buffer=tshm.buf)
                self._type_managers.append(tshm)
                self._types.append(type_array)

            else:

                #path array is float
                path_array = np.ndarray((length,), dtype=np.float32, buffer=shm.buf)

                #add a placeholder array so that later levels will have their type array index align with their data index
                self._types.append([]) 

            #we do not need to set the initial shape value
            #because we do not own the array, this should alrady be set

            #save that array
            self._arrays.append(path_array)
    
    #create a new tree from a shape
    #JBC: 2020-11-05 - reviewed for packed index use
    def create(self, shape, namespace = None, filename = None):

        #are we creating in memory or on disk
        ondisk = (filename != None)

        #if we currently have a namespace
        if self.namespace != None: self.unload()

        #this is our namespace
        #and we are the creator
        self.namespace = namespace
        self.creator = True

        #clear current data structure
        self._arrays = []

        #if in a shared namespace, we need to save off the shape
        #so that other namespaces can access us with just our name
        if namespace != None:

            #if this fails, we assume it fails
            #because the namespace already exists
            #so we immediately switch to attaching to it
            try:

                #allocate shaed memory for shape array
                self._shape_shm = mem.SharedMemory(
                        create=True, 
                        size=SymmetricTree.MAX_SHAPE_ARRAY_SIZE * SymmetricTree.NP_DATUM_SIZE,
                        name = "{}_s".format(namespace)
                )

                #log that we created the tree
                print("Created Symmetric Tree Namespace " + namespace)

            except:

                #log that we created the tree
                print("Attached to Symmetric Tree Namespace " + namespace)

                #the memory already exists
                #change to _load_from routine
                self.namespace = None
                self.creator = False
                self._load_from_namespace(namespace)
                return

            #if we got this far, the namespace did not exist and we allocated memory successfully

            #create shape array from that memory
            self._shape = np.ndarray((SymmetricTree.MAX_SHAPE_ARRAY_SIZE,), dtype=np.uint32, buffer=self._shape_shm.buf)

        
        elif ondisk:

            #create a new shape array on disk (overwriting whatever was there before)
            self._shape = np.lib.format.open_memmap("{}.shape.npy".format(filename),mode="w+",dtype=np.uint32, shape=(SymmetricTree.MAX_SHAPE_ARRAY_SIZE,))
            
        else:

            #create a new shape array without a shared memory buffer reference
            self._shape = np.ndarray((SymmetricTree.MAX_SHAPE_ARRAY_SIZE,), dtype=np.uint32)


        #if the shape passed is a list, load one way
        #but if its an ndarray - its been loaded from a file
        if type(shape) == list: 

            #set length
            self._shape[-1] = len(shape)

            #the shape (may) contain tuples with lengths and an int value defining the type
            for c in range(len(shape)):
                self._shape[c * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_SHAPE] = shape[c][0] #array shape - how many items at a single level
                self._shape[c * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TYPE] = shape[c][1] #array type (int, float, etc)
                self._shape[c * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL] = shape[c][2] * shape[c][0] #total available space for array
                self._shape[c * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_USED] = 0 #initial used size of array is ZERO

            #set filled values and zeros for remaining values
            #self._shape[:len(shape)] = shape[:]
            #self._shape[len(shape):-1] = 0

        else: 

            #copy in place as both are numpy
            self._shape[:] = shape[:]

        #step through all sizess in the shape
        #and create an array of max size to hold that shape
        #plus a few additional values to help manage the array
        for sx in range(self._shape[-1]):

            #get level info
            (s,dtype,length,used) = self.levelinfo(sx)

            #if we are creating in a shared memory namespace
            #we need to allocate that here and set the buffer
            buffer = None
            if namespace != None:

                #allocate memory for data array (path or float)
                shm = mem.SharedMemory(
                    create=True, 
                    size=int(length) * SymmetricTree.NP_DATUM_SIZE,
                    name = "{}_{}".format(namespace,sx)
                )

                #only allocate memory for type array (just bytes hince the *1) if uint type
                if dtype == 0:
                    #allocate memory, save manager and buffer
                    tshm = mem.SharedMemory(create=True, size=int(length) * 1,name = "{}_{}t".format(namespace,sx))
                    self._type_managers.append(tshm)
                    tbuffer = tshm.buf
                else:
                    #add a None for type managers so all type arrays align with data arrays
                    self._type_managers.append(None)

                #save reference to the memory manager and buffer for all data arrays
                self._managers.append(shm)
                buffer = shm.buf

            #create the array of appropriate type (type is defined in 2nd tuple item)
            if dtype == 0:

                #set the actual dtype
                np_type = np.uint32

            else:

                #set the actual dtype
                np_type = np.float32

                #there is no type array here
                type_array = []

            #if we are creating on disk, do that here
            if ondisk:

                #create our path array on disk
                path_array = np.lib.format.open_memmap("{}.{}.npy".format(filename,sx),mode="w+",dtype=np_type, shape=(length,))

                #create our type array if int type
                if dtype == 0: type_array = np.lib.format.open_memmap("{}.{}t.npy".format(filename,sx),mode="w+",dtype=np.uint8, shape=(length,))

            elif namespace != None:

                #create our path array in shared memory buffer
                path_array = np.ndarray((length,), dtype=np_type, buffer=buffer)

                #create type array if an int type
                if dtype == 0: type_array = np.ndarray((length,), dtype=np.uint8, buffer=tbuffer)

            else:

                #here we are creating not in shared memory
                path_array = np.ndarray((lengh,),dtype=np_type)

                #create the type array if int type (path)
                if dtype == 0: type_array = np.ndarray((length,),dtype=np.uint8)

            #set its initialize size and shape values
            #JBC 10-21-20 : no longer storing size in data arrays
            #path_array[-2] = s

            #save that array and typpe array references
            self._arrays.append(path_array)
            self._types.append(type_array)

        #the first array in our tree defaults to size 1 (because there is only 1 level at the top)
        self._shape[0 + SymmetricTree.LOC_USED] = 1

    #allocate space for a new index
    #JBC: 2020-11-05 - reviewed for packed index use
    def _alloc_path(self, path):

        #get the array referenced by the path
        arr = self._arrays[path[1]]

        #get level info
        (shape,_,total,used) = self.levelinfo(path[1])

        #have we exceeding maximum array size for this array?
        assert ( used + 1 ) * shape <= total, "Cannot allocate space for path {} - MAX_INDIVIDUAL_ARRAY_SIZE exceeded".format(path)

        #is this new path going to overflow the storage value in our array?
        assert (used + 1 ) <= 4294967295 , "Cannot allocate space for path {} - new size value {}+1 will overflow maxint".format(path,used)

        #add a new item to the array
        self._shape[path[1] * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_USED] += 1

        #return that index (multiplied by segment size for this array)
        return self._shape[path[1] * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_USED]

    #set a path
    #JBC: 2020-11-05 - reviewed for packed index use
    def set(self, path, value = None, mathop = None):

        #step through the path
        index = 0
        dtype = 0
        not_found = False
        for px in range(len(path)):
            #get the path tuple
            p = path[px]

            #get the array reference for this path tuple
            #the "type" is actually just the array we are using
            arr = self._arrays[p[1]]
            typ = self._types[p[1]]
            arr_size = self._shape[p[1] * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_SHAPE]

            #JBC: 2020-11-05 - moved shape from index (packed) to its own array
            ##index contains both index and which shape type we are
            #index_shape = index % self._shape[-1]
            #index = index // self._shape[-1]

            #calculate real index in array for this path
            real_index =  (index-1) * arr_size + p[0]

            #make sure real index does not overflow array size
            if real_index > self._shape[p[1] * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL]:

                #this is an issue - save the array
                #and end
                print("Error, Path {} - Index for {} <{}> Overflow".format(path,p,real_index))
                print("Saving symmetric tree to temporary file path and aborting...")
                self.save("OVERFLOWN_TREE")
                assert False, "Index over flow"

            #if we are on the last path, just set the value
            #we are done
            if px == len(path)-1:

                #allocate path (just to track the leaf array size)
                #if not_found: self._alloc_path(p)

                #figure out second index - note if we are updating
                #just one value, this will be the same as our first index
                #which is fine
                if type(value) in (list,np.ndarray):

                    #handle different math operators
                    if mathop == None:
                        arr[real_index:real_index+len(value)] = value
                    elif mathop == SymmetricTree.MATH_ADD:
                        arr[real_index:real_index+len(value)] += value
                    elif mathop == SymmetricTree.MATH_MULT:
                        arr[real_index:real_index+len(value)] *= value
                    elif mathop == SymmetricTree.MATH_DIV:
                        arr[real_index:real_index+len(value)] /= value
                    elif mathop == SymmetricTree.MATH_SUB:
                        arr[real_index:real_index+len(value)] -= value

                else:

                    #handle different math operators
                    if mathop == None:
                        arr[real_index] = value
                    elif mathop == SymmetricTree.MATH_ADD:
                        arr[real_index] += value
                    elif mathop == SymmetricTree.MATH_MULT:
                        arr[real_index] *= value
                    elif mathop == SymmetricTree.MATH_DIV:
                        arr[real_index] /= value
                    elif mathop == SymmetricTree.MATH_SUB:
                        arr[real_index] -= value

                #we are done
                return

            else:
            
                #if that value is not set, alloc
                #and update our path
                if arr[real_index] == 0: 

                    #update the value of data and value of type (each in their own arrays)
                    arr[real_index] = self._alloc_path(path[px+1])
                    typ[real_index] = path[px+1][1]
                    not_found = True

                #now get the next index (we don't care about the type here)
                index = arr[real_index]


    #unpack a path (no longer unpacking value here)
    def unpack(self, path):

        #if the incoming path is None return default
        if path == []: return (None,0,self._shape[0],0)

        #get the value and shape
        (index,index_shape) = self.get(path,include_type = True)

        #if the value was not found, return default
        if index == None: return (None, 0, self._shape[0],0)

        #JBC: 2020-11-05 - no longer packing shape into index
        #index_shape = index % self._shape[-1]
        index_size = self._shape[index_shape * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_SHAPE]
        index_start = ( index - 1 ) * index_size

        #return that as a tuple
        return (
            index,
            index_start,  #starting index of the array (adjusted as real index in array)
            index_size, #size of the array
            index_shape #shape of the index -> what array it belongs to
        )
                
    #get child paths given a path
    def children(self, path = []) :
        #locate parent if path is defined
        child_value, starting_index, child_length, child_shape = self.unpack(path)
        
        #get all children
        children = []
        for i in range(starting_index,starting_index+child_length):
            if self._arrays[child_shape][i] != 0: 
                children.append(
                    (i - starting_index, child_shape) + 
                    self.unpack(path + [(i - starting_index, child_shape)])
                )

        #return that list
        return children

    #get child paths given a path
    def child(self, path, child_index) :
        #locate parent if path is defined
        child_value, starting_index, child_length, child_shape = self.unpack(path)

        #if the child value returned from unpack is ZERO, this is not a path that exists
        #so we should not try to unpack it
        if child_value == 0: return None

        #get child
        child = (child_index , child_shape) + self.unpack(path + [(child_index, child_shape)])

        #if the first part of child path is ZERO and length is negative, this next item is not real
        if child[2] == 0: return None

        #return just the child at this index
        return child

    #internal - get a path - return a single item or range, or default.  optionally set the default value if not found
    #JBC: if include_type = True then a tuple is return with the first member being the data and second being type
    def get(self, path, default=None, items=1, set_default = False, include_type = False):

        #if there is no path, return immediately
        if len(path) == 0:
            if include_type:
                return (default,0)
            else:
                return default

        #step through the path
        not_found = False
        index = 0 #root path only ever has 1 item in it so index starts at zero
        index_shape = 0 #path root is always 0
        for px in range(len(path)):
            #get the path tuple
            p = path[px]

            #get the array reference for this path tuple
            #the "type" is actually just the array we are using
            arr = self._arrays[p[1]]
            typ = self._types[p[1]]
            arr_size = self._shape[p[1] * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_SHAPE]

            #JBC: 2020-11-05 - no longer packing type into the index value
            #index contains both index and which shape type we are
            #first strip shape and check that it matches path
            #if it doesn't, there is a problem
            #index_shape = index % self._shape[-1]

            #check that the stored array is the path array
            #if not - there is a problem
            if px > 0:
                if index_shape != p[1]:
                    print("")
                    print("Path: {}... Mismatched Shape at Leaf. Expected {} Found {}".format(
                        path[:px],
                        p[1],
                        index_shape
                    ))

            #JBC: 2020-11-05 - no longer storing type in index
            #calculate real index in array for this path
            #index = index // self._shape[-1]
            real_index = (index-1) * arr_size + p[0]

            #if we are on the last path, just set the value
            #we are done
            if px == len(path)-1:

                #if it wasn't found, update to default
                if not_found: 
                    #set default only if it's not none
                    if default != None:

                        #if we are not saving, just return default
                        if not set_default: return default

                        #set an array value from default
                        if type(default) in (list,np.ndarray):
                            #incoming deafult is an array, set it at location
                            arr[real_index:real_index+len(default)] = default

                        elif items > 1:

                            #income item is a single value but "items" length is passed
                            #so set alll items in the array
                            arr[real_index:real_index+items] = default

                        else:

                            #we are getting / setting a single value
                            arr[real_index] = default

                #if we are getting a single value just return it
                #if we are getting a range, return the range
                if items <= 1:

                    #get value at index, array reference, index, and type)
                    #if we are returning type long with data
                    if include_type:
                        return ( arr[real_index] , typ[real_index] )
                    else: 
                        return arr[real_index]

                else:
                    #get a range at index -> note that we are effectively returning a reference
                    #so the caller needs to handle that appropriately
                    return arr[real_index:real_index+items]

            else:

                #did we somehow overflow the tree
                if real_index > self._shape[path[px][1] * SymmetricTree.SHAPE_SIZE + SymmetricTree.LOC_TOTAL]:
                    print("")
                    print("Path {}: Index Overflow".format(path[:px]))
            
                #if that value is not set, alloc
                #and update our path
                if arr[real_index] == 0 and ( default == None or not set_default):
                    #we either don't need to set the default value (just return it)
                    #or there is no default to set (so don't waste time iterating the tree)
                    #so just return it
                    return default

                elif arr[real_index] == 0:

                    #there is nothing to return, but make a path
                    #because we are going to set the default
                    #JBC: 2020-11-05 - no longer packing type
                    arr[real_index] = self._alloc_path(path[px+1]) #* self._shape[-1] + path[px+1][1]
                    typ[real_index] = path[px+1][1]

                    not_found = True
                    
                #now get the next index
                index = arr[real_index]
                index_shape = typ[real_index]