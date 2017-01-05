import sys,os,argparse

execfile("CaloDNN/Arguments.py")
from keras.callbacks import EarlyStopping

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Load the Data
from CaloDNN.LoadData import *

if Mode=="Regression":
    Binning=False
if Mode=="Classification":
    Binning=[NBins,M_min,M_max,Sigma]

#InputFile="/home/afarbin/LCD/DLTools/LCD-Electrons-Pi0.h5"
InputFile="/scratch/data-backup/afarbin/LCD/LCD-Electrons-Pi0.h5"

(Train_X, Train_Y, Train_YT),(Test_X, Test_Y, Test_YT)=LoadData(InputFile,FractionTest,MaxEvents,Classification=False)

#(Train_X, Train_Y),(Test_X, Test_Y) = LoadData(InputFile,FractionTest,MaxEvents=MaxEvents)

# Normalize the Data... seems to be critical!
Norm=np.max(Train_X)
Train_X=Train_X/Norm
Test_X=Test_X/Norm

# Build/Load the Model
from DLTools.ModelWrapper import ModelWrapper
from CaloDNN.Classification import *

TXS= Train_X.shape

NInputs=TXS[1]*TXS[2]*TXS[3]

if LoadModel:
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/":
        LoadModel=LoadModel[:-1]
    Name=os.path.basename(LoadModel)
    MyModel=ModelWrapper(Name)
    MyModel.InDir=os.path.dirname(LoadModel)
    MyModel.Load()
else:
    print "Building Model...",
    sys.stdout.flush()
    MyModel=Fully3DImageClassification(Name,TXS,Width,Depth,BatchSize,200,WeightInitialization)

    # Build it
    MyModel.Build()
    print " Done."

# Print out the Model Summary
MyModel.Model.summary()

# Compile The Model
print "Compiling Model."
MyModel.Compile(Loss=loss,Optimizer=optimizer) 

# Train
if Train:
    print "Training."
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min') ]
    callbacks=[]

    MyModel.Train(Train_X, Train_Y, Epochs, BatchSize, Callbacks=callbacks)
    score = MyModel.Model.evaluate(Test_X, Test_Y, batch_size=BatchSize)

    print "Final Score:", score

# Save Model
if Train:
    MyModel.Save()

# Analysis
if Analyze:
    # ROC curve... not useful here:
    #from CaloDNN.Analysis import MultiClassificationAnalysis
    #result=MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize )
    from CaloDNN.Analysis import *
    ClassificationAnalysis(MyModel,Test_X,Test_Y,Test_YT,M_min,M_max,NBins,BatchSize)



