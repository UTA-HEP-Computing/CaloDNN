import h5py
import numpy as np
from keras.utils import np_utils
import threading

import sys
import time

#______________________________________________________________________________
def LoadData(filename, FractionTest=.1, MaxEvents=-1, Classification=True):

    F = h5py.File(filename,"r")

    X_In_Shape = F["images"].shape
    N = X_In_Shape[0]
    
    if MaxEvents>0:
        X_In = F["images"][:MaxEvents]
        Y_In = F["OneHot"][:MaxEvents]
        YT_In = F["Index"][:MaxEvents]
        N=MaxEvents
    else:
        X_In = F["images"]
        Y_In = F["OneHot"]
        YT_In = F["Index"]

    N_Test = int(round(FractionTest*N))
    N_Train = N-N_Test
        
    Train_X = X_In[:N_Train]
    Train_Y = Y_In[:N_Train]
    Train_TY = YT_In[:N_Train]

    Test_X = X_In[N_Train:]
    Test_Y = Y_In[N_Train:]
    Test_TY = YT_In[N_Train:]

    if Classification:
        Test_Y = np.sum(Test_Y.reshape(N_Test,2,100),axis=2)
        Train_Y = np.sum(Train_Y.reshape(N_Train,2,100),axis=2)
        
    return (Train_X, Train_Y,  Train_TY), (Test_X, Test_Y, Test_TY)


#______________________________________________________________________________
class XRanger:
    def __init__(self,start,stop,step=1):
        self.XR=iter(xrange(start,stop,step))
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        # acquire/release the lock when updating self.i
        with self.lock:
            return self.XR.next()

#______________________________________________________________________________
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


#______________________________________________________________________________
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


#______________________________________________________________________________
@threadsafe_generator
def LoadDataGen(filename, Classification=True, BatchSize=1024, Skip=0, Max=-1):

    F = h5py.File(filename,"r")

    X_In_Shape = F["images"].shape
    N = X_In_Shape[0]
    if Max < 0.0 or Max > N:
        Max = N

    Gen = XRanger(Skip, Max, BatchSize)

    i=0
    while True: 

        for index in Gen:
            i+=1

            X_In = F["images"][index:index+BatchSize]
            Y_In = F["OneHot"][index:index+BatchSize]

            if Classification:
                Y_In = np.sum(Y_In.reshape(BatchSize, 2, 100), axis=2)

            # hardcoded normalization
            Norm = 150 # HARDCODED
            X_In = X_In/Norm

            print threading.currentThread().getName(), i, index
#            print Y_In

            yield X_In, Y_In


#______________________________________________________________________________
def run(g, nthreads=10):
    """Starts multiple threads to execute the given function multiple
    times in each thread.
    """
    threads = [threading.Thread(target=loop, args=(g, 10)) for i in xrange(nthreads)]

    # start threads
    for t in threads:
        print "Starting", t.getName()
        t.start()

    for t in threads:
        t.join()
        print t.getName(), "Finished"


#______________________________________________________________________________
def loop(g, nmax=-1):
    i_count = 0
    for x, y in g:
        time.sleep(2)
        i_count += 1
        if nmax > 0 and i_count >= nmax:
            return


#______________________________________________________________________________
if __name__ == '__main__':

#    InputFile="/scratch/data-backup/afarbin/LCD/LCD-Electrons-Pi0.h5"

#    F = h5py.File(InputFile,"r")
#    (x,y,z),(xx,yy,zz)=LoadData(InputFile,.1,100)
#    (x,y,z),(xx,yy,zz) = LoadDataGen("/scratch/data-backup/afarbin/LCD/LCD-Electrons-Pi0.h5")

    InputFile="/scratch/data-backup/afarbin/LCD/LCD-Electrons-Pi0.h5"
    NSamples = 90000
    NTestSamples = 10000
    BatchSize = 1024
    Train_gen = LoadDataGen(InputFile, BatchSize=BatchSize, Max=NSamples)
#    run(Train_gen.next, nthreads=2)
    run(Train_gen, nthreads=2)
    



