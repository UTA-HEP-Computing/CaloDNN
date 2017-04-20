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
OutputBase="TrainedModels"
if TestMode:
    MaxEvents=int(20e3)
    NTestSamples=int(20e2)
    Epochs=10
    OutputBase+=".Test"
    print "Test Mode: Set MaxEvents to",MaxEvents,"and Epochs to", Epochs

if LowMemMode:
    n_threads=1
    multiplier=1
    
# Calculate how many events will be used for training/validation.
NSamples=MaxEvents-NTestSamples

    
# Function to help manage optional configurations. Checks and returns
# if an object is in current scope. Return default value if not.
def TestDefaultParam(Config):
    def TestParamPrime(param,default=False):
        if param in Config:
            return eval(param)
        else:
            return default
    return TestParamPrime

TestDefaultParam=TestDefaultParam(dir())

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

# (Note 1 and 2 can be made automatic with a bit of effort, but
#  will be slow due to serial writing, unless it's parallelized.)
if Premix:
    print "Using PremixGenerator."
    Train_genC = MakePreMixGenerator(InputFile, BatchSize=BatchSize,
                                     Max=NSamples, Norms=Norms,
                                     ECAL=ECAL, HCAL=HCAL,
                                     n_threads=n_threads,multiplier=multiplier,
                                     catchsignals=UseGracefulExit,
                                     sharedmemory=not LowMemMode)

    Test_genC = MakePreMixGenerator(InputFile, BatchSize=BatchSize,
                                    Skip=NSamples, Max=NTestSamples,
                                    Norms=Norms,
                                    ECAL=ECAL,multiplier=multiplier,
                                    HCAL=HCAL, n_threads=n_threads,
                                    catchsignals=UseGracefulExit,
                                    sharedmemory=not LowMemMode)
else:
    print "Using MixingGenerator."
    Train_genC = MakeMixingGenerator(FileSearch, BatchSize=BatchSize,
                                     Max=NSamples, Norms=Norms,
                                     ECAL=ECAL, HCAL=HCAL,
                                     n_threads=n_threads,multiplier=multiplier,
                                     catchsignals=UseGracefulExit)
    Test_genC = MakeMixingGenerator(FileSearch, BatchSize=BatchSize,
                                    Skip=NSamples, Max=NTestSamples,
                                    Norms=Norms,
                                    ECAL=ECAL,multiplier=multiplier,
                                    HCAL=HCAL, n_threads=n_threads,
                                    catchsignals=UseGracefulExit)

Train_gen=Train_genC.Generator()
Test_gen=Test_genC.Generator()

if Preload:
    print "Keeping data in memory after first Epoch. Hope you have a lot of memory."
    Train_gen=Train_genC.PreloadGenerator()
    Test_gen=Test_genC.PreloadGenerator()
    
# This should not be hardwired... open first file and pullout shapes?
ECALShape= None, 25, 25, 25
HCALShape= None, 5, 5, 60

# Build/Load the Model
from DLTools.ModelWrapper import ModelWrapper
from CaloDNN.Models import *

# You can automatically load the latest previous training of this model.
if TestDefaultParam("LoadPreviousModel") and not LoadModel and BuildModel:
    print "Looking for Previous Model to load."
    ModelName=Name
    if ECAL and HCAL:
        ModelName+="_Merged"
    MyModel=ModelWrapper(Name=ModelName, LoadPrevious=True,OutputBase=OutputBase)

# You can load a previous model using "-L" option with the model directory.
if LoadModel and BuildModel:    
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/": LoadModel=LoadModel[:-1]
    MyModel=ModelWrapper(Name=os.path.basename(LoadModel),InDir=os.path.dirname(LoadModel),
                         OutputBase=OutputBase)
    MyModel.Load(LoadModel)

if BuildModel and not MyModel.Model:
    FailedLoad=True
else:
    FailedLoad=False

# Or Build the model from scratch
if BuildModel and not MyModel.Model :
    import keras
    print "Building Model...",

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
        HCALModel=Fully3DImageClassification(Name+"HCAL", HCALShape, ECALWidth, HCALDepth,
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
        MyModel=MergerModel(Name+"_Merged",[ECALModel,HCALModel], NClasses, WeightInitialization,
                            OutputBase=OutputBase)

    # Configure the Optimizer, using optimizer configuration parameter.
    MyModel.Loss=loss
    # Build it
    MyModel.Build()
    print " Done."

if BuildModel:
    print "Output Directory:",MyModel.OutDir
    # Store the Configuration Dictionary
    MyModel.MetaData["Configuration"]=Config
    if "HyperParamSet" in dir():
        MyModel.MetaData["HyperParamSet"]=HyperParamSet

    # Print out the Model Summary
    MyModel.Model.summary()

    # Compile The Model
    print "Compiling Model."
    MyModel.BuildOptimizer(optimizer,Config)
    MyModel.Compile(Metrics=["accuracy"]) 

# Train
if Train or (RecoverMode and FailedLoad):
    print "Training."
    # Setup Callbacks
    # These are all optional.
    from DLTools.CallBacks import TimeStopping, GracefulExit
    from keras.callbacks import *
    callbacks=[ ]

    # Still testing this...

    if TestDefaultParam("UseGracefulExit",0):
        print "Adding GracefulExit Callback."
        callbacks.append( GracefulExit() )

    if TestDefaultParam("ModelCheckpoint",False):
        MyModel.MakeOutputDir()
        callbacks.append(ModelCheckpoint(MyModel.OutDir+"/Checkpoint.Weights.h5",
                                         monitor=TestDefaultParam("monitor","val_loss"), 
                                         save_best_only=TestDefaultParam("ModelCheckpoint_save_best_only"),
                                         save_weights_only=TestDefaultParam("ModelCheckpoint_save_weights_only"),
                                         mode=TestDefaultParam("ModelCheckpoint_mode","auto"),
                                         period=TestDefaultParam("ModelCheckpoint_period",1),
                                         verbose=0))

    if TestDefaultParam("EarlyStopping"):
        callbacks.append(keras.callbacks.EarlyStopping(monitor=TestDefaultParam("monitor","val_loss"), 
                                                       min_delta=TestDefaultParam("EarlyStopping_min_delta",0.01),
                                                       patience=TestDefaultParam("EarlyStopping_patience"),
                                                       mode=TestDefaultParam("EarlyStopping_mode",'auto'),
                                                       verbose=0))


    if TestDefaultParam("RunningTime"):
        print "Setting Runningtime to",RunningTime,"."
        TSCB=TimeStopping(TestDefaultParam("RunningTime",3600*6),verbose=False)
        callbacks.append(TSCB)
    

    # Don't fill the log files with progress bar.
    if sys.flags.interactive:
        verbose=1
    else:
        verbose=1 # Set to 2

    print "Evaluating score on test sample..."
    score = MyModel.Model.evaluate_generator(Test_gen, steps=NTestSamples/BatchSize)
    
    print "Initial Score:", score
    MyModel.MetaData["InitialScore"]=score
        
    MyModel.History = MyModel.Model.fit_generator(Train_gen,
                                                  steps_per_epoch=(NSamples/BatchSize),
                                                  epochs=Epochs,
                                                  verbose=verbose, 
                                                  validation_data=Test_gen,
                                                  validation_steps=NTestSamples/BatchSize,
                                                  callbacks=callbacks)

    score = MyModel.Model.evaluate_generator(Test_gen, steps=NTestSamples/BatchSize)


    print "Evaluating score on test sample..."
    print "Final Score:", score
    MyModel.MetaData["FinalScore"]=score

    if TestDefaultParam("RunningTime"):
        MyModel.MetaData["EpochTime"]=TSCB.history

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
    if Premix:
        Test_genC = MakePreMixGenerator(InputFile, BatchSize=BatchSize,
                                        Skip=NSamples, Max=NTestSamples,
                                        Norms=Norms, ECAL=ECAL,
                                        HCAL=HCAL, n_threads=n_threads,
                                        catchsignals=UseGracefulExit)
    else:
        Test_genC = MakeMixingGenerator(FileSearch, BatchSize=BatchSize,
                                        Skip=NSamples, Max=NTestSamples,
                                        Norms=Norms, ECAL=ECAL,
                                        HCAL=HCAL, n_threads=n_threads,
                                        catchsignals=UseGracefulExit)

    Test_genC.PreloadData()
    Test_X_ECAL, Test_X_HCAL, Test_Y = tuple(Test_genC.D)

    from DLAnalysis.Classification import MultiClassificationAnalysis
    result,NewMetaData=MultiClassificationAnalysis(MyModel,[Test_X_ECAL,Test_X_HCAL],Test_Y,BatchSize,PDFFileName="ROC",
                                                   IndexMap={0:'Pi0', 2:'ChPi', 3:'Gamma', 1:'Ele'})

    MyModel.MetaData.update(NewMetaData)
    
    # Save again, in case Analysis put anything into the Model MetaData
    if not sys.flags.interactive:
        MyModel.Save()
    else:
        print "Warning: Interactive Mode. Use MyModel.Save() to save Analysis Results."
        
# Make sure all of the Generators processes and threads are dead.
# Not necessary... but ensures a graceful exit.
# if not sys.flags.interactive:
#     for g in GeneratorClasses:
#         try:
#             g.StopFiller()
#             g.StopWorkers()
#         except:
#             pass
