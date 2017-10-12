from DLTools.ThreadedGenerator import DLMultiClassGenerator
from DLTools.ThreadedGenerator import DLh5FileGenerator
import h5py
import glob,os,sys
import random
from time import time
import numpy as np
# from data_provider_core.data_providers import H5FileDataProvider
from adlkit.data_provider import H5FileDataProvider
import math
    
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
            if Norm!=0.:
                if type(Norm) == float:
                    Ds[i]/=Norm
                if type(Norm) == str and Norm.lower()=="nonlinear" :
                    Ds[i] = np.tanh(np.sign(Ds[i]) * np.log(np.abs(Ds[i]) + 1.0) / 2.0)
                out.append(Ds[i])
        return out
    return NormalizationFunction

def RegENormalization(Norms):
    def NormalizationFunction(Ds):
        Ds[0]=Ds[0]/Norms[0]
        Ds[1]=Ds[1]/Norms[1]
        Ds[2]=Ds[2][:,1]
        return Ds
    return NormalizationFunction

# PreMix Generator.
def MergeAndNormInputs(NormFunc):
    def f(X):
        X=NormFunc(X)
        return [X[0],X[1]],X[2:]
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

        if ParticleName in Particles:
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
            SampleI[i]=EndI

    return out

def SetupData(FileSearch,
              ECAL,HCAL,target,
              NClasses,f,Particles,
              BatchSize,
              multiplier,
              ECALShape,
              HCALShape,
              ECALNorm,
              HCALNorm,
              delivery_function,
              n_threads,
              NTrain,
              NTest):
    datasets=[]
    shapes=[]
    Norms=[]

    if ECAL:
        datasets.append("ECAL")
        shapes.append((BatchSize*multiplier,)+ECALShape[1:])
        Norms.append(ECALNorm)
    if HCAL:
        datasets.append("HCAL")
        shapes.append((BatchSize*multiplier,)+HCALShape[1:])
        Norms.append(HCALNorm)
    if target:
        datasets.append("target")
#        shapes.append((BatchSize*multiplier,)+(1,5))
        shapes.append((BatchSize*multiplier,)+(2,))
        Norms.append(1.)

    # This is for the OneHot    
    shapes.append((BatchSize*multiplier, NClasses))
    Norms.append(1.)

    TrainSampleList,TestSampleList=DivideFiles(FileSearch,f,
                                               datasetnames=datasets,
                                               Particles=Particles)
    sample_spec_train = list()
    for item in TrainSampleList:
        sample_spec_train.append((item[0], item[1] , item[2], 1))

    sample_spec_test = list()
    for item in TestSampleList:
        sample_spec_test.append((item[0], item[1] , item[2], 1))

    q_multipler = 2
    read_multiplier = 1
    n_buckets = 1

    Train_genC = H5FileDataProvider(sample_spec_train,
                                    max=math.ceil(float(NTrain)/BatchSize),
                                    batch_size=BatchSize,
                                    process_function=LCDN(Norms),
                                    delivery_function=delivery_function,
                                    n_readers=n_threads,
                                    q_multipler=q_multipler,
                                    n_buckets=n_buckets,
                                    read_multiplier=read_multiplier,
                                    make_one_hot=True,
                                    sleep_duration=1,
                                    wrap_examples=True)

    Test_genC = H5FileDataProvider(sample_spec_test,
                                   max=math.ceil(float(NTest)/BatchSize),
                                   batch_size=BatchSize,
                                   process_function=LCDN(Norms),
                                   delivery_function=delivery_function,
                                   n_readers=n_threads,
                                   q_multipler=q_multipler,
                                   n_buckets=n_buckets,
                                   read_multiplier=read_multiplier,
                                   make_one_hot=True,
                                   sleep_duration=1,
                                   wrap_examples=False)

    print "Class Index Map:", Train_genC.config.class_index_map

    return Train_genC,Test_genC,Norms,shapes,TrainSampleList,TestSampleList

def MakeGenerator(ECAL,HCAL,
                  SampleList,NSamples,NormalizationFunction,
                  Merge=True,**kwargs):
    if ECAL and HCAL and Merge:
        post_f=MergeInputs()
    else:
        post_f=False
        
    pre_f=NormalizationFunction

    return DLMultiClassGenerator(SampleList, max=NSamples,
                                 preprocessfunction=pre_f,
                                 postprocessfunction=post_f,
                                 **kwargs)

def lcd_3Ddata():
    f = h5py.File("EGshuffled.h5", "r")
    data = f.get('ECAL')
    dtag = f.get('TAG')
    xtr = np.array(data)
    tag = np.array(dtag)
    # xtr=xtr[...,numpy.newaxis]
    # xtr=numpy.rollaxis(xtr,4,1)
    print xtr.shape

    return xtr, tag.astype(bool)


# ##################################################################################################################
# CaloDNN
# ##################################################################################################################


def LCDN(Norms):
    def NormalizationFunction(Ds):
        # converting the data from an ordered-dictionary format to a list
        Ds = [Ds[item] for item in Ds]
        out = []
        # print('DS', Ds)
        # TODO replace with zip function
        for i,D in enumerate(Ds):
            Norm=Norms[i]
            if Norm != 0.:
                if isinstance(Norm, float):
                    D /= Norm
                if isinstance(Norm, str) and Norm.lower() == "nonlinear":
                    D = np.tanh(
                        np.sign(Ds[i]) * np.log(np.abs(Ds[i]) + 1.0) / 2.0)
                out.append(D)
        return out

    return NormalizationFunction

def unpack(thing):
    #print('thing', '([{0}, {1}], {2})'.format(thing[0].shape, thing[1].shape, thing[2].shape)) 
    return [thing[:2], thing[2]]
