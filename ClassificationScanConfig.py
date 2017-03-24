import random
import getopt
from DLTools.Permutator import *
import sys,argparse
from numpy import arange
import os

# Input for Premixed Generator
# Find first existing instance
PossibleInputFile= ["/data/afarbin/LCD/LCD-Merged-All.h5",
                    "/Users/afarbin/LCD/Data/LCD-Merged-All.h5"]
try:
    InputFile=filter( os.path.isfile, PossibleInputFile )[0]
except:
    print "Warning: no inputfile found in",PossibleInputFile

# Input for Mixing Generator
FileSearch="/data/afarbin/LCD/*/*.h5"
#FileSearch="/Users/afarbin/LCD/Data/*/*.h5"

# Generation Model
Config={
    "MaxEvents":int(3.e6),
    "NTestSamples":100000,
    "NClasses":4,

    "Epochs":1000,
    "BatchSize":1024,

    # Configures the parallel data generator that read the input.
    # These have been optimized by hand. Your system may have
    # more optimal configuration.
    "n_threads":4,  # Number of workers
    "multiplier":2, # Read N batches worth of data in each worker

    # How weights are initialized
    "WeightInitialization":"'normal'",

    # Normalization determined by hand.
    "ECAL":True,
    "ECALNorm":150.,

    # Normalization needs to be determined by hand. 
    "HCAL":True,
    "HCALNorm":150.,

    # Set the ECAL/HCAL Width/Depth for the Dense model.
    # Note that ECAL/HCAL Width/Depth are changed to "Width" and "Depth",
    # if these parameters are set. 
    "HCALWidth":32,
    "HCALDepth":2,
    "ECALWidth":32,
    "ECALDepth":2,

    # No specific reason to pick these. Needs study.
    # Note that the optimizer name should be the class name (https://keras.io/optimizers/)
    "loss":"'categorical_crossentropy'",

    "activation":"'relu'",
    "BatchNormalization":True,
    "Dropout":False,
    
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
    "RunningTime": 6*3600,

    # Load last trained version of this model configuration. (based on Name var below)
    "LoadPreviousModel":True
}

# Parameters to scan and their scan points.
Params={ "Width":[32,64,128,256,512],
         "Depth":range(1,5),
         "lr":[0.1,0.01],
         "decay":[0.1,0.01],
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
