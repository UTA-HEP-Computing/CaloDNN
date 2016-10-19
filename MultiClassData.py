import numpy as np
import h5py
from keras.utils import np_utils
import sys
from time import time

def shuffle_in_unison_inplace(a, b, c=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if type(c) != bool:
        return a[p], b[p], c[p]
    return a[p], b[p]

def OrganizeFiles(Samples, OpenFiles=False):
    Files={}

    NFiles=len(Samples)
    index=0
    for S in Samples:
        if len(S)==2:
            ClassName=DataSetName=S[1]
            File=S[0]

        if len(S)==3:
            DataSetName=S[1]
            ClassName=S[2]
            File=S[0]

        if not ClassName in Files.keys():
            Files[ClassName]={ "N":0,
                               "Files":[],
                               "File_I":0,
                               "Example_I":0,
                               "DataSetName":DataSetName,
                               "ClassIndex":index} 
            index+=1
        
        if OpenFiles:
            print "Opening",index,"/",NFiles,":" ,S
            sys.stdout.flush()
            try:
                f=h5py.File(File)
            except:
                print
                print "Failed Opening:",S
                continue

            N=np.shape(f[DataSetName])[0]
            Files[ClassName]["N"]+=N
            f.close()

        Files[ClassName]["Files"].append(File)


    return Files


def MultiClassGenerator(Samples,batchsize, verbose=True, 
                        OneHot=True, ClassIndex=False, Energy=False, ClassIndexMap=False):
    Classes=OrganizeFiles(Samples)
    N_Classes=len(Classes.keys())

    N_ExamplePerClass=int(batchsize/N_Classes)
    remainEx=batchsize-N_Classes*N_ExamplePerClass
    if verbose:
        print "Found",N_Classes," classes. Will pull", N_ExamplePerClass," examples from each class."
        print "Will have", remainEx, "remaining.. will randomly pad classes."
        
    myClassIndexMap={}
    ClassNames=Classes.keys()

    while True:
        X=False
        Y=False
        first=True

        for C in Classes:
            Classes[C]["NExamples"]=N_ExamplePerClass

        for i in xrange(0,int(remainEx)):
            ii=int(float(N_Classes)*np.random.random())
            #        print "Padding ",ClassNames[ii]
            Classes[ClassNames[ii]]["NExamples"]+=1
        
        count=0
        N_TotalExamples=0
        for C in Classes:

            Cl=Classes[C]
            myClassIndexMap[C]=Cl["ClassIndex"]

            myEnergy=float(C.split("_")[-1])
            count+=1

            N_Examples=0
            while N_Examples<Cl["NExamples"]:
                if Cl["File_I"] >= len(Cl["Files"]):
                    print "Warning out of files for ",C
                    break

                if verbose:
#                    print Cl["File_I"], Cl["Example_I"], N_Examples, len(Cl["Files"])
                    print count,"/",N_Classes,":",C,":", Cl["Files"][Cl["File_I"]], 

                start=time()
                if Cl["Example_I"]==0:
                    if "f" in dir():
                        f.close()
                    f=h5py.File(Cl["Files"][Cl["File_I"]])
#                    Data=Cl["Data"]=np.array(f[Cl["DataSetName"]])
#                else:
#                   Data=Cl["Data"]
                if verbose:
                    print "t=",time()-start, "Find File.",

                N=f[Cl["DataSetName"]].shape[0] #Data.shape[0]
                I=Cl["Example_I"]
                N_Unused=N-I
                N_End=min(I+(Cl["NExamples"]-N_Examples), N)
                N_Using=N_End-I

#                if verbose:
#                    print N, I, N_Unused, N_End, N_Using

                concat=False

                start=time()
                if first:
                    first=False
                    finalShape= f[Cl["DataSetName"]].shape 
                    finalShape=(batchsize,)+finalShape[1:]
                    X=np.zeros(finalShape)
                    Y=np.zeros(batchsize)
                    if Energy:
                        E=np.zeros(batchsize)

                a=np.empty(N_Using); a.fill(Cl["ClassIndex"])
                if Energy:
                    b=np.empty(N_Using); b.fill(myEnergy)

                if verbose:
                    print "Adding",N_Using," to",  N_TotalExamples," Events.",
                X[N_TotalExamples:N_TotalExamples+N_Using]=f[Cl["DataSetName"]][I:N_End] 
                Y[N_TotalExamples:N_TotalExamples+N_Using]=a
                if Energy:
                    E[N_TotalExamples:N_TotalExamples+N_Using]=b
                    
                if verbose:
                    print "t=",time()-start, "Concatenate."

                N_Examples+=N_Using
                N_TotalExamples+=N_Using
                if N_End >= N:
                    Cl["Example_I"]=0
                    Cl["File_I"]+=1
                else:
                    Cl["Example_I"]=N_End



        if verbose:
            print "Shuffling."
        start=time()
        if Energy:
            X,Y,E=shuffle_in_unison_inplace(X,Y,E)
        else:
            X,Y=shuffle_in_unison_inplace(X,Y)
        Y1= np_utils.to_categorical(Y)
        if verbose:
            print "t=",time()-start, "Shuffle."

        results = (X,)
        if OneHot:
            results += (Y1,)

        if ClassIndex:
            results += (Y,)

        if Energy:
            results += (E,)
            
        if ClassIndexMap:
            results+=(myClassIndexMap,)

        yield results



def LoadMultiClassData(Samples, FractionTest=.1, MaxEvents=-1, MinEvents=-1):
#    FHandles={}
    Data={}
    ClassIndex={}

    NFiles=len(Samples)
    # Get Data Out of every file
    index=0
    for S in Samples:
        if len(S)==2:
            ClassName=DataSetName=S[1]
            File=S[0]

        if len(S)==3:
            DataSetName=S[1]
            ClassName=S[2]
            File=S[0]

        print "Opening",index,"/",NFiles,":" ,S,
        sys.stdout.flush()
        try:
            f=h5py.File(File)
        except:
            N=-1
            print
            print "Failed Opening:",S
            
        if ClassName in Data:
            print "Found ",np.shape(f[DataSetName])[0], "Events in file. ",
            Data[ClassName]=np.concatenate((Data[ClassName], np.array(f[DataSetName])))
                
        else:
            Data[ClassName]=np.array(f[DataSetName])
        N=np.shape(Data[ClassName])[0]

        print N," Events in class."
#        FHandles[ClassName]=f

        ClassIndex[ClassName]=index

        index+=1
        f.close()

    # MergeData and Create Labels

    First=True

    Train_X=None
    Train_Y=None

    Test_X=None
    Test_Y=None


    for S in Data:
        N=np.shape(Data[S])[0]
        N_Test=int(round(FractionTest*N))
        N_Train=N-N_Test

        if MaxEvents!=-1:
            if MaxEvents>N:
                print "Warning: Sample",S," has",N," events which is less that ",MaxEvents,"."
                print "Using ",NTrain,"Events for training."
                print "Using ",NTest,"Events for training."
            else:
                N_Test=int(round(FractionTest*MaxEvents))
                N_Train=MaxEvents-N_Test

        if MinEvents!=-1:
            if N_Train<MinEvents:
                print "Warning: Sample",S," has",N_Train," training events which is less that ",MaxEvents,"."

        if not First:
            Train_X=np.concatenate((Train_X,Data[S][0:N_Train]))
            a=np.empty(N_Train); a.fill(ClassIndex[S])
            Train_Y=np.concatenate((Train_Y,a))

            Test_X=np.concatenate((Test_X,Data[S][-N_Test:]))
            a=np.empty(N_Test); a.fill(ClassIndex[S])
            Test_Y=np.concatenate((Test_Y,a))
        else:
            Train_X=Data[S][0:N_Train]
            a=np.empty(N_Train); a.fill(ClassIndex[S])
            Train_Y=a

            Test_X=Data[S][-N_Test:]
            a=np.empty(N_Test); a.fill(ClassIndex[S])
            Test_Y=a
            First=False        

        # Random Shuffle
     
    Train_X,Train_Y=shuffle_in_unison_inplace(Train_X,Train_Y)
    Test_X,Test_Y=shuffle_in_unison_inplace(Test_X,Test_Y)
        
    Train_Y= np_utils.to_categorical(Train_Y)
    Test_Y= np_utils.to_categorical(Test_Y)

    return (Train_X, Train_Y), (Test_X, Test_Y), ClassIndex



