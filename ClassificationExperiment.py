import sys,os,argparse

# Parse the Arguments
execfile("CaloDNN/ClassificationArguments.py")

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Use "--Test" to run on less events and epochs.
if TestMode:
    MaxEvents=int(2e4)
    Epochs=3
    print "Test Mode: Set MaxEvents to",MaxEvents," and Epochs to", Epochs

# Calculate how many events will be used for training/validation.
NTestSamples=int(FractionTest*MaxEvents)
NSamples=MaxEvents-NTestSamples

# Load the Data
from CaloDNN.LCDData import * 

# We apply a constant Normalization to the input and target data.
# The ConstantNormalization function, which is applied during reading,
# takes a list with the entries corresponding to the Tensors read from
# input file. For LCD Data set the tensors are [ ECAL, HCAL, OneHot ]
# so the normalization constant is [ ECALNorm, HCALNorm, 1. ]. 

Norms=[]
if ECAL: Norms.append(ECALNorm)
if HCAL: Norms.append(HCALNorm)
Norms.append(1.)

# We have 3 methods of reading the LCD Data set, which comes in as a
# large set of files separated into subdirectories corresponding to
# particle type. For training, this data needs to be mixed and
# "OneHot" labels created. Everything uses the ThreadedGenerator from
# DLTools.
#
# Methods (from slow to fast):
# 1. Read and mix the files on fly. (--nopremix)
# 2. Premix the files into single input file using LCDData.py. 
# read the large file on the fly. (default)
# 3. Load the data into memory, so Epochs>1 are significantly
# accelerated. Uses a lot of memory. Works either 1 or 2 
# above. (--preload)
#
# (Note 1 and 2 can be made automatic with a bit of effort, but
#  will be slow due to serial writing, unless it's parallelized.)
if Premix:
    print "Using PremixGenerator."
    Train_genC = MakePreMixGenerator(InputFile, BatchSize=BatchSize, Max=NSamples,
                                     Norms=Norms, ECAL=ECAL, HCAL=HCAL, n_threads=n_threads)
    Test_genC  = MakePreMixGenerator(InputFile, BatchSize=BatchSize, Skip=NSamples, Max=NTestSamples,
                                     Norms=Norms, ECAL=ECAL, HCAL=HCAL, n_threads=n_threads)
else:
    print "Using MixingGenerator."
    Train_genC = MakeMixingGenerator(FileSearch, BatchSize=BatchSize, Max=NSamples,
                                     Norms=Norms, ECAL=ECAL, HCAL=HCAL, n_threads=n_threads)
    Test_genC  = MakeMixingGenerator(FileSearch, BatchSize=BatchSize, Skip=NSamples, Max=NTestSamples,
                                     Norms=Norms, ECAL=ECAL, HCAL=HCAL, n_threads=n_threads)

Train_gen=Train_genC.Generator()
Test_gen=Test_genC.Generator()

if Preload:
    print "Keeping data in memory after first Epoch."
    Train_gen=Train_genC.PreloadGenerator()

# We need data in memory for analysis.
# If your FractionTest is high, you may run out of memory.
if Preload or Analyze: 
    Test_gen=Test_genC.PreloadGenerator()
    
# This should not be hardwired... open first file and pullout shapes?
ECALShape= BatchSize, 25, 25, 25
HCALShape= BatchSize, 5, 5, 60

# Build/Load the Model
from DLTools.ModelWrapper import ModelWrapper
from CaloDNN.Models import *

# You can load a previous model using "-L" option with the model directory.
if LoadModel: 
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/": LoadModel=LoadModel[:-1]
    MyModel=ModelWrapper(Name=os.path.basename(LoadModel),InDir=os.path.dirname(LoadModel))
    MyModel.Load(LoadModel)
else:
    import keras
    print "Building Model...",
        
    if ECAL:
        ECALModel=Fully3DImageClassification(Name+"ECAL", ECALShape, ECALWidth, ECALDepth,
                                             BatchSize, NClasses, WeightInitialization)
        ECALModel.Build()
        MyModel=ECALModel

    if HCAL:
        HCALModel=Fully3DImageClassification(Name+"HCAL", HCALShape, ECALWidth, HCALDepth,
                                             BatchSize, NClasses, WeightInitialization)
        HCALModel.Build()
        MyModel=HCALModel

    if HCAL and ECAL:
        MyModel=MergerModel(Name+"_Merged",[ECALModel,HCALModel], NClasses, WeightInitialization)

    # Configure the Optimizer, using optimizer configuration parameter.
    MyModel.BuildOptimizer(optimizer,Config)
    MyModel.Loss=loss
    # Build it
    MyModel.Build()
    print " Done."

# Store the Configuration Dictionary
MyModel.MetaData["Configuration"]=Config

# Print out the Model Summary
MyModel.Model.summary()

# Compile The Model
print "Compiling Model."
MyModel.Compile() 

# Function to help manage optional configurations. Checks and returns
# if an object is in current scope. Return default value if not.
def TestParam(param,default=False):
    if param in dir():
        return eval(param)
    else:
        return default

# Train
if Train:
    print "Training."

    # Setup Callbacks
    # These are all optional.
    from DLTools.CallBacks import TimeStopping
    from keras.callbacks import *
    callbacks=[]

    if TestParam("ModelCheckpoint"):
        MyModel.MakeOutputDir()
        callbacks.append(ModelCheckpoint(MyModel.OutDir+"/Checkpoint.Weights.h5",
                                         monitor=TestParam("monitor","val_loss"), 
                                         save_best_only=TestParam("ModelCheckpoint_save_best_only"),
                                         save_weights_only=TestParam("ModelCheckpoint_save_weights_only"),
                                         mode=TestParam("ModelCheckpoint_mode","auto"),
                                         period=TestParam("ModelCheckpoint_period",1),
                                         verbose=0))

    if TestParam("EarlyStopping"):
        callbacks.append(keras.callbacks.EarlyStopping(monitor=TestParam("monitor","val_loss"), 
                                                       min_delta=TestParam("EarlyStopping_min_delta",0.01),
                                                       patience=TestParam("EarlyStopping_patience"),
                                                       mode=TestParam("EarlyStopping_mode",'auto'),
                                                       verbose=0))


    if TestParam("RunningTime"):
        print "Setting Runningtime to",args.runningtime,"."
        callbacks.append(TimeStopping(TestParam(RunningTime),verbose=False))
    

    # Don't fill the log files with progress bar.
    if sys.flags.interactive:
        verbose=1
    else:
        verbose=3
        
    MyModel.Model.fit_generator(Train_gen,
                                validation_data=Test_gen,
                                nb_val_samples=NTestSamples,
                                nb_epoch=Epochs,
                                samples_per_epoch=NSamples,
                                callbacks=callbacks,
                                verbose=verbose, 
                                nb_worker=1,
                                pickle_safe=False)

    print "Evaluating score on test sample..."
    score = MyModel.Model.evaluate_generator(Test_gen,
                                             val_samples=NTestSamples, 
                                             nb_worker=1,
                                             pickle_safe=False)
    print "Done."
    print "Final Score:", score

    # Store the parameters used for scanning for easier tables later:
    for k in Params:
        MyModel.MetaData[k]=Config[k]
    # Save Model
    MyModel.Save()
else:
    print "Skipping Training."
    
# Analysis
if Analyze:
    print "Running Analysis."
    Test_X_ECAL, Test_X_HCAL, Test_Y = tuple(Test_genC.D)

    from CaloDNN.Analysis import MultiClassificationAnalysis
    result=MultiClassificationAnalysis(MyModel,[Test_X_ECAL,Test_X_HCAL],Test_Y,BatchSize,
                                       IndexMap={0:'Pi0', 2:'ChPi', 3:'Gamma', 1:'Ele'})

    # Save again, in case Analysis put anything into the Model MetaData
    MyModel.Save()

# Make sure all of the Generators processes and threads are dead.
# Not necessary... but ensures a graceful exit.
if not sys.flags.interactive:
    for g in GeneratorClasses:
        try:
            g.StopFiller()
            g.StopWorkers()
        except:
            pass
