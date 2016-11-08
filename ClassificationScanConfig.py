import random
import getopt
from DLTools.Permutator import *
import sys,argparse

# Generation Model
Config={
    "GenerationModel":"'Load'",
    "MaxEvents":1e5,
    "FractionTest":0.1,
    "NClasses":2,
    
    "M_min":0,
    "M_max":200,

    "Sigma":0.,

    "Epochs":100,
    "BatchSize":2048,
    
    "LearningRate":0.005,
    
    "Decay":0.,
    "Momentum":0.,
    "Nesterov":0.,

    "WeightInitialization":"'normal'",

    "Mode":"'Classification'",
    "NBins":200,

    "loss":"'categorical_crossentropy'",
    "optimizer":"'rmsprop'"

}

Params={ "Width":[32,64,128,256,512],
         "Depth":range(1,10),
          }

PS=Permutator(Params)
Combos=PS.Permutations()

print "HyperParameter Scan: ", len(Combos), "possible combiniations."

if "HyperParamSet" in dir():
    i=int(HyperParamSet)
else:
# Set Seed based on time
    random.seed()
    i=int(round(len(Combos)*random.random()))
    print "Randomly picking HyperParameter Set"

print "Picked combination: ",i

for k in Combos[i]:
    Config[k]=Combos[i][k]

Name="MEDNN"

for MetaData in Params.keys():
    val=str(Config[MetaData]).replace('"',"")
    Name+="_"+val.replace("'","")

print "Model Filename: ",Name

# Possibilties for future reference
WeightInitializations=[
    "uniform",
    "lecun_uniform",
    "normal",
    "identity",
    "orthogonal",
    "zero",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform"]

Losses=[
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "squared_hinge",
    "hinge",
    "binary_crossentropy",
    "categorical_crossentropy",
    "poisson",
    "cosine_proximity"]
