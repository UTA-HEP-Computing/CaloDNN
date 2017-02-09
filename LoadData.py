import h5py
import numpy as np
from keras.utils import np_utils
import threading

import sys
import time

#______________________________________________________________________________
class XRanger:
    def __init__(self,start,stop,step=1):
        self.XR=iter(xrange(start,stop,step))
        self.lock = threading.Lock()
        self.start=start
        self.stop=stop
        self.step=step
        self.i=start
        
    def __iter__(self):
        return self

    def next(self):
        # acquire/release the lock when updating self.i
        with self.lock:
            self.i+=self.step
            if self.i<self.stop:
                return self.i
            
            #return self.XR.next()

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
def ConstantNormalization(Norms):
    def NormalizationFunction(Ds):
        out = []
        for i,D in enumerate(Ds):
            D/=Norms[i]
            out.append(D)
        
        return out
    return NormalizationFunction

#______________________________________________________________________________
@threadsafe_generator
def LoadDataGen(filename,  datasets=["ECAL","HCAL","OneHot"],
                BatchSize=1024, Skip=0, Max=-1,
                Normalization=False, Wrap=True, verbose=False):

    F = h5py.File(filename,"r")

    X_In_Shape = F[datasets[0]].shape
    N = X_In_Shape[0]

    if Max < 0.0 or Max > N:
        Max = N

    Gen = XRanger(Skip, Max, BatchSize)

    i=0
    Done=False
    while not Done: 
        for index in Gen:
            i+=1

            Ins=[]
            for D in datasets:
                Ins.append( F[D][index:index+BatchSize])

            if Normalization:
                Ins=Normalization(Ins)

            if verbose:
                print threading.currentThread().getName(), i, index

            yield tuple(Ins)

        Done = not Wrap

@threadsafe_generator
def MaskGenerator(G,mask):
    for g in G:
        res=g
        out=[]
        for i,m in enumerate(mask):
            if m:
              out.append(res[i])
        yield tuple(out)
        

## Testing Code
#______________________________________________________________________________
def run(g, nthreads=10, iterations=-1):
    """Starts multiple threads to execute the given function multiple
    times in each thread.
    """
    threads = [threading.Thread(target=loop, args=(g, iterations)) for i in xrange(nthreads)]

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
    for D in g:
        #time.sleep(2)
        for d in D:
            print d.shape
            first=d[0]
            print first[np.where(first>0)]
            
        i_count += 1
        if nmax > 0 and i_count >= nmax:
            return


#______________________________________________________________________________
if __name__ == '__main__':


    InputFile="/home/afarbin/LCD/DLKit/LCD-Merged-All.h5"
    BatchSize=1024
    
    Train_gen = MaskGenerator( LoadDataGen(InputFile, BatchSize=BatchSize, Max=-1, verbose=True,
                                           Normalization=ConstantNormalization([150.,150.,1])),
                               [1,0,1] )
    
    run(Train_gen, nthreads=2)
    



