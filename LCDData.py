from MultiClassData import *
import glob,os,sys

from time import time


#______________________________________________________________________________
def LoadData(FileSearch="/scratch/data-backup/afarbin/LCD/*/*.h5", FractionTest=0.1, MaxEvents=-1, MaxFiles=-1):
    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    Samples=[]

    FileCount=0

    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        Energy=float(basename.split("_")[-1].replace("GeV.h5",""))

        Samples.append((F,"images",ParticleName+"_"+str(Energy)))
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break

    (Train_X, Train_Y), (Test_X, Test_Y), ClassIndex=LoadMultiClassData(Samples,.1,MaxEvents=MaxEvents)

    return  (Train_X, Train_Y), (Test_X, Test_Y), ClassIndex


#(Train_X, Train_Y), (Test_X, Test_Y), ClassIndex=LoadData(MaxFiles=100)


#______________________________________________________________________________
def LCDDataGenerator(batchsize=2048,FileSearch="/home/afarbin/LCD/Data/*/*.h5",MaxFiles=-1,
                     verbose=True, OneHot=True, ClassIndex=False, Energy=False, ClassIndexMap=False):
    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    Samples=[]

    FileCount=0

    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        Energy=float(basename.split("_")[-1].replace("GeV.h5",""))

        Samples.append((F,"images",ParticleName+"_"+str(Energy)))
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break


    return MultiClassGenerator(Samples,batchsize,
                               verbose=verbose, 
                               OneHot=OneHot,
                               ClassIndex=ClassIndex, 
                               Energy=Energy, 
                               ClassIndexMap=ClassIndexMap)


def MergeData(filename, NEvents=2e6, batchsize=2**17,verbose=True):
    g=LCDDataGenerator(batchsize,
                       verbose=verbose, 
                       OneHot=True, ClassIndex=True, Energy=True, ClassIndexMap=True)
    f=h5py.File(filename,"w")

    first=True

    NEvents=int(NEvents)
    batchsize=int(batchsize)
    chunksize=2048

    for i in range(0,NEvents,batchsize):
        startT=time()
        D=next(g)
        
        if first:
            first=False
            shape=(NEvents,)+D[0].shape[1:]
            chunk=(chunksize,)+D[0].shape[1:]
            dsetX = f.create_dataset("images",shape,compression="gzip", chunks=chunk)
            
            shape=(NEvents,)+D[1].shape[1:]
            chunk=(chunksize,)+D[1].shape[1:]
            dsetY = f.create_dataset("OneHot",shape,compression="gzip", chunks=chunk)

            shape=(NEvents,)+D[2].shape[1:]
            chunk=(chunksize,)+D[2].shape[1:]
            dsetY1 = f.create_dataset("Index",shape,compression="gzip", chunks=chunk)

            shape=(NEvents,)+D[3].shape[1:]
            chunk=(chunksize,)+D[3].shape[1:]
            dsetE = f.create_dataset("Energy",shape,compression="gzip", chunks=chunk)


        dsetX[i:i+batchsize]=D[0]
        dsetY[i:i+batchsize]=D[1]
        dsetY1[i:i+batchsize]=D[2]
        dsetE[i:i+batchsize]=D[3]

        if verbose:
            print "t=",time()-startT, " Batch Creation."
            print "Shapes:"
            print dsetX.shape
            print dsetY.shape
            print dsetY1.shape
            print dsetE.shape
        
            print "Class Index Dictionary."
            print D[4]

    f.close()
    return D[4]

MergeData("LCD-Electrons-Pi0-2.h5")
