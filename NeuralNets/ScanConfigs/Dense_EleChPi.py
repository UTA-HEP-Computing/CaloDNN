###################
# Helper Function #
###################

def TestDefaultParam(Config):
    def TestParamPrime(param,default=False):
        if param in Config:
            return eval(param)
        else:
            return default
    return TestParamPrime

# TestDefaultParam("param") returns either value of param or default
TestDefaultParam=TestDefaultParam(dir())

##########################
# Configuration Settings #
##########################

import random
import getopt
from DLTools.Permutator import *
import sys,argparse
from numpy import arange
import os

from multiprocessing import cpu_count
from DLTools.Utils import gpu_count

# Save location
saveFolder = "/home/mazhang/DLKit/CaloDNN/NeuralNets/Cache/Dense_EleChPi_50Epochs/"
if not os.path.exists(os.path.dirname(saveFolder)):
    os.makedirs(os.path.dirname(saveFolder))

# Number of threads
max_threads=12
n_gpu=gpu_count()
if n_gpu>0:
    n_threads=int(min(round(cpu_count()/n_gpu),max_threads))
else:
    n_threads=max(1,cpu_count()-2)
print "Found",cpu_count(),"CPUs and",gpu_count(),"GPUs. Using",n_threads,"threads. max_threads =",max_threads

# Particle types
Particles=["ChPi","Ele"]

# ECAL shapes (add dimensions for conv net)
ECALShape= None, 25, 25, 25
HCALShape= None, 5, 5, 60

# Input for mixing generator
FileSearch="/data/LCD/V2/EleChPi/*/*.h5"

# Config settings (to save)
Config={
    "MaxEvents":int(3.e5),
    "NTestSamples":int(3.e5 * 0.2),
    "NClasses":len(Particles),

    "Epochs":50,
    "BatchSize":1024,

    # Configures the parallel data generator that read the input.
    # These have been optimized by hand. Your system may have
    # more optimal configuration.
    "n_threads":n_threads, #n_threads,  # Number of workers
    "n_threads_cache":5,
    "multiplier":1, # Read N batches worth of data in each worker

    # How weights are initialized
    "WeightInitialization":"'normal'",

    # Normalization determined by hand.
    "ECAL":True,
    "ECALNorm":"'NonLinear'",

    # Normalization needs to be determined by hand. 
    "HCAL":True,
    "HCALNorm":"'NonLinear'",

    # Set the ECAL/HCAL Width/Depth for the Dense model.
    # Note that ECAL/HCAL Width/Depth are changed to "Width" and "Depth",
    # if these parameters are set. 
    "HCALWidth":32,
    "HCALDepth":2,
    "ECALWidth":64,
    "ECALDepth":2,

    # No specific reason to pick these. Needs study.
    # Note that the optimizer name should be the class name (https://keras.io/optimizers/)
    "loss":"'categorical_crossentropy'",

    "activation":"'relu'",
    "BatchNormLayers":True,
    "DropoutLayers":True,
    
    # Specify the optimizer class name as True (see: https://keras.io/optimizers/)
    # and parameters (using constructor keywords as parameter name).
    # Note if parameter is not specified, default values are used.
    "optimizer":"'RMSprop'",
    "lr":0.01,    
    "decay":0.001,

    # Parameter monitored by Callbacks
    "monitor":"'val_loss'",

    # Active Callbacks
    # Specify the CallBack class name as True (see: https://keras.io/callbacks/)
    # and parameters (using constructor keywords as parameter name,
    # with classname added).
    "ModelCheckpoint":True,
    "Model_Chekpoint_save_best_only":False,    

    # Configure Running time callback
    # Set RunningTime to a value to stop training after N seconds.
    "RunningTime": 24*3600,

    # Load last trained version of this model configuration. (based on Name var below)
    "LoadPreviousModel":True
}

# Add config settings to current scope
for a in Config:
    exec(a+"="+str(Config[a]))

###################
# Hyperparameters #
###################

# Parameters to scan and their scan points.
Params={"optimizer":["'RMSprop'","'Adam'","'SGD'"],
        "Width":[32,64,128,256,512],
        "Depth":range(1,5),
        "lr":[0.01,0.001],
        "decay":[0.01,0.001],
        }

# Get all possible configurations.
PS=Permutator(Params)
Combos=PS.Permutations()
print "HyperParameter Scan: ", len(Combos), "possible combiniations."

# HyperParameter sets are numbered. You can iterate through them using
# the -s option followed by an integer .
i=0
if "HyperParamSet" in dir():
    i=int(HyperParamSet)

for k in Combos[i]: Config[k]=Combos[i][k]

# Use the same Width and/or Depth for ECAL/HCAL if these parameters 
# "Width" and/or "Depth" are set.
if "Width" in Config:
    Config["ECALWidth"]=Config["Width"]
    Config["HCALWidth"]=Config["Width"]
if "Depth" in Config:
    Config["ECALDepth"]=Config["Depth"]
    Config["HCALDepth"]=Config["Depth"]

# Build a name for the this configuration using the parameters we are
# scanning.
Name="CaloDNN"
for MetaData in Params.keys():
    val=str(Config[MetaData]).replace('"',"")
    Name+="_"+val.replace("'","")

if "HyperParamSet" in dir():
    print "______________________________________"
    print "ScanConfiguration"
    print "______________________________________"
    print "Picked combination: ",i
    print "Combo["+str(i)+"]="+str(Combos[i])
    print "Model Filename: ",Name
    print "______________________________________"
else:
    for ii,c in enumerate(Combos):
        print "Combo["+str(ii)+"]="+str(c)

################
# Create Model #
################

from CaloDNN.NeuralNets.Models import *
trainCache = saveFolder + "Train.h5"
testCache = saveFolder + "Test.h5"
OutputBase = saveFolder + "Model" # Save folder

if ECAL:
    ECALModel=Fully3DImageClassification(Name+"ECAL", ECALShape, ECALWidth, ECALDepth,
					 BatchSize, NClasses,
					 init=TestDefaultParam("WeightInitialization",'normal'),
					 activation=TestDefaultParam("activation","relu"),
					 Dropout=TestDefaultParam("DropoutLayers",0.5),
					 BatchNormalization=TestDefaultParam("BatchNormLayers",False),
					 NoClassificationLayer=ECAL and HCAL,
					 OutputBase=OutputBase)
    ECALModel.Build()
    MyModel=ECALModel

if HCAL:
    HCALModel=Fully3DImageClassification(Name+"HCAL", HCALShape, HCALWidth, HCALDepth,
					 BatchSize, NClasses,
					 init=TestDefaultParam("WeightInitialization",'normal'),
					 activation=TestDefaultParam("activation","relu"),
					 Dropout=TestDefaultParam("DropoutLayers",0.5),
					 BatchNormalization=TestDefaultParam("BatchNormLayers",False),
					 NoClassificationLayer=ECAL and HCAL,
					 OutputBase=OutputBase)
    HCALModel.Build()
    MyModel=HCALModel

if HCAL and ECAL:
    ConfigModel=MergerModel(Name+"_Merged",[ECALModel,HCALModel], NClasses, WeightInitialization,
			OutputBase=OutputBase)

ConfigModel.Loss=loss # Configure the Optimizer, using optimizer configuration parameter.
