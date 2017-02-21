## Reads in the LCD h5 files from Maurizio. Merge, assign labels, and other data prep necessary for running models.

#from MultiClassData import *
from DLTools.ThreadedGenerator import DLMultiClassGenerator

import glob,os,sys
import random
from time import time

def LCDDataGenerator(datasetnames,batchsize=2048,FileSearch="/data/afarbin/LCD/*/*.h5",MaxFiles=-1,
                     verbose=True, OneHot=True, ClassIndex=False, ClassIndexMap=False,n_threads=4,multiplier=1,timing=False):
    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    if MaxFiles!=-1:
        random.shuffle(Files)
    Samples=[]

    FileCount=0

    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")

        Samples.append((F,datasetnames,ParticleName))
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break

    GC= DLMultiClassGenerator(Samples,batchsize,
                              verbose=verbose, 
                              #OneHot=OneHot,
                              ClassIndex=ClassIndex,
                              n_threads=n_threads,
                              multiplier=multiplier,
                              timing=timing)

            
    if ClassIndexMap:
        return [GC,GC.ClassIndexMap]
    else:
        return GC

def MergeData(filename, h5keys=["ECAL","HCAL","target"], NEvents=1e8, batchsize=2048,verbose=True, MaxFiles=-1):
    # Create a generator
    
    [GenClass,IndexMap]=LCDDataGenerator(h5keys, batchsize,
                                         verbose=verbose, 
                                         OneHot=True, ClassIndex=True, ClassIndexMap=True, MaxFiles=MaxFiles)

    g=GenClass.Generator()
    
    f=h5py.File(filename,"w")

    first=True

    NEvents=int(NEvents)
    batchsize=int(batchsize)
    chunksize=2048

    myh5keys=h5keys+["index","OneHot"]

    IndexMap=None
    
    dsets=[]
    i=0
    for D in g:
        if i>=NEvents:
            break
        
        startT=time()
        
        if first:
            first=False
            for j,T in enumerate(D[:-1]):
                shape=(batchsize,)+T.shape[1:]
                maxshape=(NEvents,)+T.shape[1:]
                chunk=(chunksize,)+T.shape[1:]
                dsets.append( f.create_dataset(myh5keys[j],shape,compression="gzip", chunks=chunk, maxshape=maxshape))

        for j,T in enumerate(D[:-1]):
            # Extend the dataset
            shape=dsets[j].shape
            dsets[j].resize((shape[0]+batchsize,)+shape[1:])
            # Store the data
            dsets[j][i:i+batchsize]=T
            
        if verbose:
            print "Batch Creation time =",time()-startT
            sys.stdout.flush()

        i+=batchsize

    f.close()

    return IndexMap

if __name__ == '__main__':
    IndexMap=MergeData("LCD-Merged-All.h5",batchsize=2**16,NEvents=1e8)
#    IndexMap=MergeData("LCD-Merged-All.h5",batchsize=2**11,NEvents=1e5,MaxFiles=10)
    print "IndexMap: ", IndexMap
