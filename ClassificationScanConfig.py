import random
import getopt
from DLTools.Permutator import *
import sys,argparse

# Input for Premixed Generator
InputFile="/home/afarbin/LCD/DLKit/LCD-Merged-All.h5"
# Input for Mixing Generator
FileSearch="/data/afarbin/LCD/*/*.h5"

# Generation Model
Config={
    "GenerationModel":"'Load'",
    "MaxEvents":int(.5e6),
    "FractionTest":0.1,
    "NClasses":4,

    "M_min":0,
    "M_max":200,

    "Sigma":0.,

    "Epochs":100,
    "BatchSize":1024,

    # Configures the parallel data generator that read the input.
    # These have been optimized by hand. Your system may have
    # more optimal configuration.
    "n_threads":4,  # Number of workers
    "multiplier":2, # Read N batches worth of data in each worker

    "WeightInitialization":"'normal'",

    "Mode":"'Classification'",

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
    "optimizer":"'RMSprop'",

    # Place holders
    # These DO NOT WORK.
    "LearningRate":0.005,
    "Decay":0.,
    "Momentum":0.,
    "Nesterov":0.,
}

# Parameters to scan and their scan points.
Params={ "Width":[32,64,128,256,512],
         "Depth":range(1,10),
          }

# Get all possible configurations.
PS=Permutator(Params)
Combos=PS.Permutations()
print "HyperParameter Scan: ", len(Combos), "possible combiniations."

# HyperParameter sets are numbered. You can iterate through them using
# the -s option followed by an integer .
if "HyperParamSet" in dir():
    i=int(HyperParamSet)
print "Picked combination: ",i
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
print "Model Filename: ",Name


