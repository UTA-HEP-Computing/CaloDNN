import sys,os,argparse

execfile("CaloDNN/ClassificationArguments.py")
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

InputFile="/home/afarbin/LCD/DLKit/LCD-Merged-All.h5"
from DLTools.ThreadedGenerator import DLh5FileGenerator

#LoadDataGen
GeneratorClasses=[]

def MakeGenerator(BatchSize,Max=-1,Skip=0):
    if useGenerator:
        G=DLh5FileGenerator(files=[InputFile], datasets=["ECAL","OneHot"],batchsize=BatchSize,
                            max=Max, skip=Skip, verbose=False, Wrap=True,
                            n_threads=n_threads,multiplier=multiplier,
                            postprocessfunction=ConstantNormalization([150.,1.]))
        GeneratorClasses.append(G)

        return G.Generator()
    else:
        # Uses the old generator when loading all data in memory, so we don't have to pipe 30 GB between processes.
        return LoadDataGen(InputFile, datasets=["ECAL","OneHot"],BatchSize=BatchSize, Max=Max,
                           Skip=Skip, verbose=False,
                           Normalization=ConstantNormalization([150.,1.]))
    
if TestMode:
    MaxEvents=int(2e4)
    Epochs=2
    print "Test Mode: Set MaxEvents to",MaxEvents," and Epochs to", Epochs

NTestSamples=int(FractionTest*MaxEvents)
NSamples=MaxEvents-NTestSamples

if useGenerator:
    print "Using Generator."
    Train_gen = MakeGenerator(BatchSize=BatchSize, Max=NSamples)
    Test_gen  = MakeGenerator(BatchSize=BatchSize, Skip=NSamples, Max=NTestSamples)
    Test2_gen = MakeGenerator(BatchSize=NTestSamples, Skip=NSamples)
else:
    print "Loading All Events in Memory... "
    print "This will take a while and take a lot of memory. But it will train fast."
    print "Use --generator if you want load events on the fly and save memory. But training is slower. "

    if Train:
        Train_gen = MakeGenerator(BatchSize=NSamples)
        Train_X, Train_Y = Train_gen.next()
    if Train or Analyze:    
        Test_gen = MakeGenerator(BatchSize=NTestSamples, Skip=NSamples)
        Test_X, Test_Y = Test_gen.next()
    
# This should not be hardwired
TXS = BatchSize, 25, 25, 25
NInputs=TXS[1]*TXS[2]*TXS[3]
print "NInputs is %i" % NInputs

# Build/Load the Model
from DLTools.ModelWrapper import ModelWrapper
from CaloDNN.Classification import *

if LoadModel: # This really needs to get cleaned up in the ModelWrapper base class.
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/":
        LoadModel=LoadModel[:-1]
    Name=os.path.basename(LoadModel)
    MyModel=ModelWrapper(Name)
    MyModel.InDir=os.path.dirname(LoadModel)
    MyModel.Load(LoadModel)
    MyModel.Initialize()
    os.mkdir(MyModel.OutDir)  
else:
    print "Building Model...",
    sys.stdout.flush()
    MyModel=Fully3DImageClassification(Name, TXS, Width, Depth, BatchSize, NClasses, WeightInitialization)

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
    #        callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min') ]
    callbacks=[]

    if useGenerator:

        MyModel.Model.fit_generator(Train_gen,
                                    nb_epoch=Epochs,
                                    nb_worker=nb_worker,
                                    #verbose=3,
                                    samples_per_epoch=NSamples,
                                    callbacks=callbacks,
                                    pickle_safe=False)

        score = MyModel.Model.evaluate_generator(Test_gen,
                                                 val_samples=NTestSamples, 
                                                 max_q_size=10,
                                                 nb_worker=nb_worker,
                                                 pickle_safe=False)

    else:
        MyModel.Train(Train_X, Train_Y, Epochs, BatchSize, Callbacks=callbacks)
        score = MyModel.Model.evaluate(Test_X, Test_Y, batch_size=BatchSize)
    
    print "Final Score:", score

    # Save Model
    MyModel.Save()

# Analysis
if Analyze:
    if useGenerator:
        Test_X, Test_Y = Test2_gen.next()

    from CaloDNN.Analysis import MultiClassificationAnalysis
    result=MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize,
                                       IndexMap={0:'Pi0', 2:'ChPi', 3:'Gamma', 1:'Ele'})


for g in GeneratorClasses:
    g.StopFiller()
    g.StopWorkers()
