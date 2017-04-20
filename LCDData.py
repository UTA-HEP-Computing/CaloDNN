# Provides 3 methods of reading the LCD Data set, which come in as a large set of files separated
# into subdirectories corresponding to particle type. For training, this data needs to be mixed
# and "OneHot" labels created. Everything uses the ThreadedGenerator from DLTools.
#
# Methods:
# 1. Read and mix the files on fly.
# 2. Premix the files into single input file. Read the large file on the fly.
# 3. Load the data into memory, so Epochs>1 are significantly accelerated... but extremely memory heavy.


from DLTools.ThreadedGenerator import DLMultiClassGenerator
from DLTools.ThreadedGenerator import DLh5FileGenerator
import h5py
import glob,os,sys
import random
from time import time
import numpy as np

GeneratorClasses=[]

def ConstantNormalization(Norms):
    def NormalizationFunction(Ds):
        out = []
        for i,Norm in enumerate(Norms):
            Ds[i]/=Norm
            out.append(Ds[i])
        return out
    return NormalizationFunction

def LCDNormalization(Norms):
    def NormalizationFunction(Ds):
        out = []
        for i,Norm in enumerate(Norms):
            if type(Norm) == float:
                Ds[i]/=Norm
            if type(Norm) == str and Norm=="NonLinear" :
                Ds[i] = np.tanh(np.sign(Ds[i]) * np.log(np.abs(Ds[i]) + 1.0) / 2.0)
            out.append(Ds[i])
        return out
    return NormalizationFunction


# PreMix Generator.
def MergeAndNormInputs(NormFunc):
    def f(X):
        X=NormFunc(X)
        return [X[0],X[1]],X[2]
    return f

def MergeInputs():
    def f(X):
        return [X[0],X[1]],X[2]
    return f

def MakePreMixGenerator(InputFile,BatchSize,Norms=[150.,1.],
                        Max=-1,Skip=0, ECAL=True, HCAL=True, Energy=False, 
                        **kwargs):
    datasets=[]
    if ECAL:
        datasets.append("ECAL")
    if HCAL:
        datasets.append("HCAL")

    datasets.append("OneHot")

    if Energy:
        datasets.append("target")
    
    if ECAL and HCAL:
        post_f=MergeInputs()
    else:
        post_f=False
        
    pre_f=LCDNormalization(Norms)
    
    G=DLh5FileGenerator(files=[InputFile], datasets=datasets,
                        batchsize=BatchSize,
                        max=Max, skip=Skip, 
                        postprocessfunction=post_f,
                        preprocessfunction=pre_f,
                        **kwargs)
    
    GeneratorClasses.append(G)

    return G

# Mix on the fly generator
def LCDDataGenerator(datasetnames,batchsize=2048,FileSearch="/data/afarbin/LCD/*/*.h5",MaxFiles=-1,
                     verbose=False, OneHot=True, ClassIndex=False, ClassIndexMap=False):
    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    #if MaxFiles!=-1:
    #    random.shuffle(Files)
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

def MakeMixingGenerator(FileSearch,BatchSize,Norms=[150.,1.], Max=-1, Skip=0,  ECAL=True, HCAL=True, Energy=False, **kwargs):

    if ECAL:
        datasets.append("ECAL")
    if HCAL:
        datasets.append("HCAL")

    if Energy:
        datasets.append("target")

    if ECAL and HCAL:
        f=MergeInputs(LCDNormalization(Norms))
    else:
        f=LCDNormalization(Norms)


    [G,IndexMap]=LCDDataGenerator(datasets, BatchSize,
                                  OneHot=False, ClassIndex=False, ClassIndexMap=True,
                                  **kwargs)

    GeneratorClasses.append(G)

    return G


##
## Uses the Mixing Generator to premix data and write out to a file.
##
##

def MergeData(FileSearch, filename, h5keys=["ECAL","HCAL","target"], NEvents=1e8, batchsize=2048,
              n_threads=4,verbose=False, MaxFiles=-1):
    # Create a generator
    
    [GenClass,IndexMap]=LCDDataGenerator(h5keys, batchsize,FileSearch=FileSearch,
                                         verbose=verbose, n_threads=n_threads,
                                         OneHot=True, ClassIndex=True, ClassIndexMap=True, MaxFiles=MaxFiles)

    g=GenClass.Generator()

    print "Writing out to ",filename
    f=h5py.File(filename,"w")

    first=True

    NEvents=int(NEvents)
    batchsize=int(batchsize)
    chunksize=2048

    myh5keys=h5keys+["index","OneHot"]

    IndexMap=None

    start=time()

    dsets=[]
    i=0
    count=1
    print "Events : Time : Avg Time/Batch : Write Time/Batch"
    for D in g:

        Delta=(time()-start)
        print i,":",Delta, ":",Delta/float(count),":",
        count+=1
        
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
            
        print time()-startT
        i+=batchsize

    f.close()

    return IndexMap

if __name__ == '__main__':
    import sys
    FileSearch="/data/afarbin/LCD/*/*.h5"

    try:
        OutFile=sys.argv[1]
    except:
        OutFile="LCD-Merged-Test.h5"

    try:
        n_threads=int(sys.argv[2])
    except:
        n_threads=4

    print "Note that we are stopping the merger at 3M events because we run out of pi0s."
    IndexMap=MergeData(FileSearch,OutFile,batchsize=2048,NEvents=3e6,n_threads=n_threads)
    print "IndexMap: ", IndexMap
