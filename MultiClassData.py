import numpy as np
import h5py
from keras.utils import np_utils
import sys
from time import time
import copy

def shuffle_in_unison_inplace(Data):
    N=len(Data[0])
    p = np.random.permutation(N)

    out = []
    for d in Data:
        assert N == len(d)
        out.append(d[p])

    return out

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


def MultiClassGenerator(Samples, batchsize, verbose=True, 
                        OneHot=True, ClassIndex=False, ClassIndexMap=False, Wrap=False):
    Classes=OrganizeFiles(Samples)
    N_Classes=len(Classes.keys())

    N_ExamplePerClass=int(batchsize/N_Classes)
    remainEx=batchsize-N_Classes*N_ExamplePerClass
    if verbose:
        print "Found",N_Classes," classes. Will pull", N_ExamplePerClass," examples from each class."
        print "Will have", remainEx, "remaining.. will randomly pad classes."
        
    myClassIndexMap={}
    ClassNames=Classes.keys()

    Data=[]
    first=True
    IndexT=None

    Done = False
    Stop = False
    while not Done:
        for C in Classes:
            Classes[C]["NExamples"]=N_ExamplePerClass

        # Randomly choose how many from each class
        for i in xrange(0,int(remainEx)):
            ii=int(float(N_Classes)*np.random.random()) # Should this be Possion?
            #        print "Padding ",ClassNames[ii]
            Classes[ClassNames[ii]]["NExamples"]+=1
        
        count=0
        N_TotalExamples=0
        
        for C in Classes:
            count+=1

            Cl=Classes[C]
            myClassIndexMap[C]=Cl["ClassIndex"]

            # Pull examples from this class
            N_Examples=0
            while N_Examples<Cl["NExamples"]:
                if Cl["File_I"] >= len(Cl["Files"]):
                    print "Warning: out of files for",C
                    Done=not Wrap
                    if not Wrap:
                        print "Stopping Generator."
                        #Stop=True
                        break
                    else:
                        print "Wrapping. Starting with first file for",C
                        Cl["File_I"]=0

                if verbose:
#                    print Cl["File_I"], Cl["Example_I"], N_Examples, len(Cl["Files"])
                    print count,"/",N_Classes,":",C,":", Cl["Files"][Cl["File_I"]],
                start=time()
                if Cl["Example_I"]==0:
                    if "File" in Cl:
                        Cl["File"].close()
                    if verbose:
                        print "Opening:",Cl["File_I"],
                    f=Cl["File"]=h5py.File(Cl["Files"][Cl["File_I"]],"r")
                else:
                    f=Cl["File"]
                    
                #if verbose:
                #    print "t=",time()-start, "Find File."

                N=f[Cl["DataSetName"][0]].shape[0] 
                I=Cl["Example_I"]
                N_Unused=N-I
                N_End=min(I+(Cl["NExamples"]-N_Examples), N)
                N_Using=N_End-I

                #if verbose:
                #    print N, I, N_Unused, N_End, N_Using

                concat=False

                start=time()

                if first:
                    first=False
                    for DataSetName in Cl["DataSetName"]:
                        # Note for try/excepts below: Try to find the dataset in the file, if not there, it must
                        # be index or something else constructed here
                        try: 
                            finalShape= f[DataSetName].shape 
                            finalShape=(batchsize,)+finalShape[1:]
                        except:
                            finalShape=(batchsize,1)                        
                        Data.append(np.zeros(finalShape))

                        # Fill class index based on samples definition.
                    IndexT=np.zeros(batchsize)

                a=np.empty(N_Using); a.fill(Cl["ClassIndex"])

                #if verbose:
                #    print "Adding",N_Using," to",  N_TotalExamples," Events.",

                for i in xrange(0,len(Cl["DataSetName"])):
                    try:
                        Data[i][N_TotalExamples:N_TotalExamples+N_Using]=f[Cl["DataSetName"][i]][I:N_End] 
                    except:
                        pass
                        
                IndexT[N_TotalExamples:N_TotalExamples+N_Using]=a

                if verbose:
                    print "t=",time()-start, "Concatenate."

                N_Examples+=N_Using
                N_TotalExamples+=N_Using
                if N_End >= N:
                    Cl["Example_I"]=0
                    Cl["File_I"]+=1
                else:
                    Cl["Example_I"]=N_End

        if not Stop:
            out=tuple(Data)
            
            if ClassIndex:
                out+=(IndexT,)

            if OneHot:
                Y1=np_utils.to_categorical(IndexT)
                out+=(Y1,)

            #if verbose:
            #    print "Shuffling."
            start=time()
            out=shuffle_in_unison_inplace(out) 

            #if verbose:
            #    print "t=",time()-start, "Shuffle."
            
            if ClassIndexMap:
                out+=(myClassIndexMap,)

            yield out



if __name__ == '__main__':
    pass
