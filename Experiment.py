import sys,os,argparse

execfile("CaloNN/Arguments.py")
from keras.callbacks import EarlyStopping

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Load the Data
from MEDNN.LoadData import *

if Mode=="Regression":
    Binning=False
if Mode=="Classification":
    Binning=[NBins,M_min,M_max,Sigma]

InputFile="/home/afarbin/LCD/DLTools/LCD-Electrons-Pi0.h5"

(Train_X, Train_Y),(Test_X, Test_Y) = LoadData(InputFile,MaxEvents=MaxEvents)

# Normalize the Data... seems to be critical!
Norm=np.max(Train_X)
Train_X=Train_X/Norm
Test_X=Test_X/Norm

# Build/Load the Model
from DLModels.Regression import *
from DLTools.ModelWrapper import ModelWrapper

# Instantiate a LSTM AutoEncoder... 

NInputs=15

if LoadModel:
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/":
        LoadModel=LoadModel[:-1]
    Name=os.path.basename(LoadModel)
    MyModel=ModelWrapper(Name)
    MyModel.InDir=os.path.dirname(LoadModel)
    MyModel.Load()
else:
    if Mode=="Regression":
        MyModel=FullyConnectedRegression(Name,NInputs,Width,Depth,WeightInitialization)
    if Mode=="Classification":
        MyModel=FullyConnectedClassification(Name,NInputs,Width,Depth,Binning[0],WeightInitialization)

    # Build it
    MyModel.Build()

# Print out the Model Summary
MyModel.Model.summary()

# Compile The Model
print "Compiling Model."
MyModel.Compile(loss=loss,optimizer=optimizer) 

# Train
if Train:
    print "Training."
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min') ]
    callbacks=[]
    if Mode=="Regression":
        MyModel.Train(Train_X, Train_Y, Epochs, BatchSize,Callbacks=callbacks)
        score = MyModel.Model.evaluate(Test_X, Test_Y, batch_size=BatchSize)

    if Mode=="Classification":
        MyModel.Train(Train_X, Train_Y, Epochs, BatchSize, Callbacks=callbacks)
        score = MyModel.Model.evaluate(Test_X, Test_Y, batch_size=BatchSize)

    print "Final Score:", score

# Save Model
if Train:
    MyModel.Save()

# Analysis
if Analyze:
    from DLAnalysis.Regression import *
    if Mode=="Regression":
        RegressionAnalysis(MyModel,Test_X,Test_Y,M_min,M_max,BatchSize)

    if Mode=="Classification":
        ClassificationAnalysis(MyModel,Test_X,Test_Y,Test_YT,M_min,M_max,NBins,BatchSize)


