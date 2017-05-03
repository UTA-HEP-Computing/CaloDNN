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
            if type(Norm) == str and Norm.lower()=="nonlinear" :
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

def DivideFiles(FileSearch="/data/LCD/*/*.h5",Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    print "Found",len(Files),"files."
    
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")

        try:
            Samples[ParticleName].append((F,datasetnames,ParticleName))
        except:
            Samples[ParticleName]=[(F,datasetnames,ParticleName)]

        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    
    out=[] 
    for j in range(len(Fractions)):
        out.append([])
        
    SampleI=len(Samples.keys())*[int(0)]
    
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)

        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI+1

    return out

