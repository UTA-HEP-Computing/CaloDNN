# Parse the Arguments
# execfile("CaloDNN/ClassificationArguments.py")
# ##############################################################################
# Arg parsing
# ##############################################################################
import argparse
import logging as lg

from DLKit.DLTools.CallBacks import GracefulExit, TimeStopping
from DLKit.DLTools.Permutator import Permutator
from keras.callbacks import *

from CaloDNN.LoadData import *
from CaloDNN.Models import *

lg.basicConfig(level=lg.WARNING)

# Configuration of this jobConfig
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',
                    default="CaloDNN/ClassificationScanConfig.py",
                    help="Use specified configuration file.")

parser.add_argument('-L', '--LoadModel',
                    help='Loads a model from specified directory.',
                    default=False)
parser.add_argument('--gpu', dest='gpuid', default="",
                    help='Use specified GPU.')
parser.add_argument('--cpu', action="store_true", help='Use CPU.')
parser.add_argument('--NoTrain', action="store_true",
                    help="Do not run training.")
parser.add_argument('--NoAnalysis', action="store_true",
                    help="Do not run analysis.")
parser.add_argument('--NoModel', action="store_true",
                    help="Do not build or load model.")
parser.add_argument('--LowMem', action="store_true",
                    help="Minimize Memory Usage.")
parser.add_argument('--Test', action="store_true",
                    help="Run in test mode (reduced examples and epochs).")
parser.add_argument('--Recover', action="store_true",
                    help="Train only if fail to load model (use with --NoTrain and --Load or --LoadPrevious).")
parser.add_argument('-s', "--hyperparamset", default="0",
                    help="Use specificed (by index) hyperparameter set.")
parser.add_argument('--nocache', action="store_true",
                    help="Do not use cache data to disk for faster read > 1st epoch.")
parser.add_argument('--preload', action="store_true",
                    help="Preload the data into memory. Caution: requires lots of memory.")
parser.add_argument('-r', "--runningtime", default="0",
                    help="End training after specified number of seconds.")
parser.add_argument('-p', "--LoadPrevious", action="store_true",
                    help="Load the last trained model.")
parser.add_argument('--GracefulExit', action="store_true",
                    help="Enable graceful exit via Ctrl-C or SIGTERM signal.")

# parser.add_argument('--generator', action="store_true")

# Configure based on commandline flags... this really needs to be cleaned up
args = parser.parse_args()
Train = not args.NoTrain
Analyze = not args.NoAnalysis
BuildModel = not args.NoModel
if not BuildModel:
    Train = False
    Analyze = False

TestMode = args.Test
RecoverMode = args.Recover
UseGPU = not args.cpu
gpuid = args.gpuid
verbose=1
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)

ConfigFile = args.config
# useGenerator = args.generator
Cache = not args.nocache
Preload = args.preload
LoadPreviousModel = args.LoadPrevious
LoadModel = args.LoadModel
LowMemMode = args.LowMem
UseGracefulExit = args.GracefulExit

if int(args.runningtime) > 0:
    RunningTime = int(args.runningtime)

# Configuration from PBS:
if "PBS_ARRAYID" in os.environ:
    HyperParamSet = int(os.environ["PBS_ARRAYID"])

if "PBS_QUEUE" in os.environ:
    if "cpu" in os.environ["PBS_QUEUE"]:
        UseGPU = False
    if "gpu" in os.environ["PBS_QUEUE"]:
        UseGPU = True
        gpuid = int(os.environ["PBS_QUEUE"][3:4])

if UseGPU:
    print "Using GPU", gpuid
    os.environ[
        'THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%s,floatX=float32,force_device=True" % (
        gpuid)
else:
    print "Using CPU."

# ##############################################################################














# Process the ConfigFile
# execfile(ConfigFile)
# ##############################################################################
# Arg parsing
# ##############################################################################



max_threads = 12
# n_threads = int(min(round(cpu_count() / gpu_count()), max_threads))

n_threads = 3

Particles = ["ChPi", "Gamma", "Pi0", "Ele"]

# Input for Mixing Generator
FileSearch = "/data/LCD/V1/*/*.h5"

# Generation Model
Config = {
    "MaxEvents": int(3.e6),
    "NTestSamples": 100000,
    "NClasses": 2,

    "Epochs": 1000,
    "BatchSize": 1024,

    # Configures the parallel data generator that read the input.
    # These have been optimized by hand. Your system may have
    # more optimal configuration.
    "n_threads": n_threads,  # Number of workers
    "n_threads_cache": 5,
    "multiplier": 1,  # Read N batches worth of data in each worker

    # How weights are initialized
    "WeightInitialization": 'random_normal',

    # Normalization determined by hand.
    "ECAL": True,
    "ECALNorm": "'NonLinear'",

    # Normalization needs to be determined by hand.
    "HCAL": True,
    "HCALNorm": "'NonLinear'",

    # Set the ECAL/HCAL Width/Depth for the Dense model.
    # Note that ECAL/HCAL Width/Depth are changed to "Width" and "Depth",
    # if these parameters are set.
    "HCALWidth": 32,
    "HCALDepth": 2,
    "ECALWidth": 32,
    "ECALDepth": 2,

    # No specific reason to pick these. Needs study.
    # Note that the optimizer name should be the class name (https://keras.io/optimizers/)
    "loss": 'categorical_crossentropy',

    "activation": "'relu'",
    "BatchNormLayers": True,
    "DropoutLayers": True,

    # Specify the optimizer class name as True (see: https://keras.io/optimizers/)
    # and parameters (using constructor keywords as parameter name).
    # Note if parameter is not specified, default values are used.
    "optimizer": 'rmsprop',
    "lr": 0.01,
    "decay": 0.001,

    # Parameter monitored by Callbacks
    "monitor": "'val_loss'",

    # Active Callbacks
    # Specify the CallBack class name as True (see: https://keras.io/callbacks/)
    # and parameters (using constructor keywords as parameter name,
    # with classname added).
    "ModelCheckpoint": True,
    "Model_Chekpoint_save_best_only": False,

    # Configure Running time callback
    # Set RunningTime to a value to stop training after N seconds.
    "RunningTime": 1 * 3600,

    # Load last trained version of this model configuration. (based on Name var below)
    "LoadPreviousModel": True
}

# Parameters to scan and their scan points.
Params = {"optimizer": ['RMSprop', 'Adam', 'SGD'],
          "Width": [32, 64, 128, 256, 512],
          "Depth": range(1, 5),
          "lr": [0.01, 0.001],
          "decay": [0.01, 0.001],
          }

# Get all possible configurations.
PS = Permutator(Params)
Combos = PS.Permutations()
print "HyperParameter Scan: ", len(Combos), "possible combiniations."

# HyperParameter sets are numbered. You can iterate through them using
# the -s option followed by an integer .
i = 0

# TODO
# check args breh
# if "HyperParamSet" in dir():
#     i = int(args.hyperparamset)

for k in Combos[i]:
    Config[k] = Combos[i][k]

# Use the same Width and/or Depth for ECAL/HCAL if these parameters
# "Width" and/or "Depth" are set.
if "Width" in Config:
    Config["ECALWidth"] = Config["Width"]
    Config["HCALWidth"] = Config["Width"]
if "Depth" in Config:
    Config["ECALDepth"] = Config["Depth"]
    Config["HCALDepth"] = Config["Depth"]

# Build a name for the this configuration using the parameters we are
# scanning.
Name = "CaloDNN"
for MetaData in Params.keys():
    val = str(Config[MetaData]).replace('"', "")
    Name += "_" + val.replace("'", "")

if "HyperParamSet" in dir():
    print "______________________________________"
    print "ScanConfiguration"
    print "______________________________________"
    print "Picked combination: ", i
    print "Combo[" + str(i) + "]=" + str(Combos[i])
    print "Model Filename: ", Name
    print "______________________________________"
else:
    for ii, c in enumerate(Combos):
        print "Combo[" + str(ii) + "]=" + str(c)

# ##############################################################################

# TODO print and frame this
# Now put config in the current scope. Must find a prettier way.
# if "Config" in dir():
#     for a in Config:
#         exec (a + "=" + str(Config[a]))


# Use "--Test" to run on less events and epochs.
OutputBase = "TrainedModels"
if TestMode:
    Config['MaxEvents'] = int(20e3)
    Config['NTestSamples'] = int(20e2)
    Config['Epochs'] = 10
    OutputBase += ".Test"
    print ("Test Mode: Set MaxEvents to ",
           Config['MaxEvents'],
           " and Epochs to ",
           Config['Epochs'])

if LowMemMode:
    n_threads = 1
    multiplier = 1

# Calculate how many events will be used for training/validation.
NSamples = Config['MaxEvents'] - Config['NTestSamples']


# Function to help manage optional configurations. Checks and returns
# if an object is in current scope. Return default value if not.
def TestDefaultParam(Config):
    def TestParamPrime(param, default=False):
        if param in Config:
            return eval(param)
        else:
            return default

    return TestParamPrime


TestDefaultParam = TestDefaultParam(dir())

# We apply a constant Normalization to the input and target data.
# The ConstantNormalization function, which is applied during reading,
# takes a list with the entries corresponding to the Tensors read from
# input file. For LCD Data set the tensors are [ ECAL, HCAL, OneHot ]
# so the normalization constant is [ ECALNorm, HCALNorm, 1. ]. 


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

# Load the Data


ECALShape = None, 25, 25, 25
HCALShape = None, 5, 5, 60
ECAL = Config['ECAL']
HCAL = Config['HCAL']

# Too many parameters, should be decomposed or use an object
TrainSampleList, TestSampleList, Norms, shapes = SetupData(FileSearch,
                                                           Config['ECAL'],
                                                           HCAL,
                                                           False,
                                                           Config['NClasses'],
                                                           [float(
                                                               NSamples) /
                                                            Config['MaxEvents'],
                                                            float(
                                                                Config[
                                                                    'NTestSamples']) /
                                                            Config[
                                                                'MaxEvents']],
                                                           Particles,
                                                           Config['BatchSize'],
                                                           Config['multiplier'],
                                                           ECALShape,
                                                           HCALShape,
                                                           Config['ECALNorm'],
                                                           Config['HCALNorm'])

sample_spec_train = list()
for item in TrainSampleList:
    sample_spec_train.append((item[0], item[1] + ['target'], item[2], 1))


sample_spec_test = list()
for item in TestSampleList:
    sample_spec_test.append((item[0], item[1] + ['target'], item[2], 1))


# ##############################################################################
# print('TrainSampleList', TrainSampleList)
# print('TestSampleList', TestSampleList)

# ##############################################################################
# # Use DLGenerators to read data
# Train_genC = MakeGenerator(ECAL, HCAL, TrainSampleList, NSamples,
#                            LCDNormalization(Norms),
#                            batchsize=BatchSize,
#                            shapes=shapes,
#                            n_threads=n_threads,
#                            multiplier=multiplier,
#                            cachefile="/tmp/CaloDNN-LCD-TrainEvent-Cache_reece.h5")
#
# Test_genC = MakeGenerator(ECAL, HCAL, TestSampleList, NTestSamples,
#                           LCDNormalization(Norms),
#                           batchsize=BatchSize,
#                           shapes=shapes,
#                           n_threads=n_threads,
#                           multiplier=multiplier,
#                           cachefile="/tmp/CaloDNN-LCD-TestEvent-Cache_reece.h5")

# print "Train Class Index Map:", Train_genC.ClassIndexMap
#
# if Preload:
#     print "Caching data in memory for faster processing after first epoch. Hope you have enough memory."
#     Train_gen = Train_genC.PreloadGenerator()
#     Test_gen = Test_genC.PreloadGenerator()
# elif Cache:
#     print "Caching data on disk for faster processing after first epoch. Hope you have enough disk space."
#     Train_gen = Train_genC.DiskCacheGenerator(n_threads_cache)
#     Test_gen = Test_genC.DiskCacheGenerator(n_threads_cache)
# else:
#     Train_gen = Train_genC.Generator()
#     Test_gen = Test_genC.Generator()

# ##############################################################################

from data_provider_core.data_providers import H5FileDataProvider
#from sample_spec import train_sample_spec, test_sample_spec

from .lcd_utils import LCDN, unpack

Train_gen = H5FileDataProvider(sample_spec_train,
                               batch_size=Config['BatchSize'],
                               process_function=LCDN(Norms),
                               delivery_function=unpack,
                               n_readers=2,
                               q_multipler=10,
                               wrap_examples=True)

Test_gen = H5FileDataProvider(sample_spec_test,
                              batch_size=Config['BatchSize'],
                              process_function=LCDN(Norms),
                              delivery_function=unpack,
                              n_readers=2,
                              q_multipler=10,
                              wrap_examples=True)

start_time = time()
lg.debug("starting generators")
Train_gen.start()
Test_gen.start()
lg.debug("start successful")
lg.info("generator_spin_up={0}".format(time() - start_time))

# Build/Load the Model

# ############
# Cleanup Instantiations
MyModel = None

# You can automatically load the latest previous training of this model.
if TestDefaultParam("LoadPreviousModel") and not LoadModel and BuildModel:
    print "Looking for Previous Model to load."
    ModelName = Name
    if ECAL and HCAL:
        ModelName += "_Merged"
    MyModel = ModelWrapper(Name=ModelName, LoadPrevious=True,
                           OutputBase=OutputBase)

# You can load a previous model using "-L" option with the model directory.
if LoadModel and BuildModel:
    print "Loading Model From:", LoadModel
    if LoadModel[-1] == "/": LoadModel = LoadModel[:-1]
    MyModel = ModelWrapper(Name=os.path.basename(LoadModel),
                           InDir=os.path.dirname(LoadModel),
                           OutputBase=OutputBase)
    MyModel.Load(LoadModel)

if BuildModel and not MyModel:
    FailedLoad = True
else:
    FailedLoad = False

# Or Build the model from scratch
if BuildModel and not MyModel:
    import keras

    print "Building Model...",

    if ECAL:
        ECALModel = Fully3DImageClassification(Name + "ECAL",
                                               ECALShape,
                                               Config['ECALWidth'],
                                               Config['ECALDepth'],
                                               Config['BatchSize'],
                                               Config['NClasses'],
                                               init=TestDefaultParam(
                                                   "WeightInitialization",
                                                   'random_normal'),
                                               activation=TestDefaultParam(
                                                   "activation", "relu"),
                                               Dropout=TestDefaultParam(
                                                   "DropoutLayers", 0.5),
                                               BatchNormalization=TestDefaultParam(
                                                   "BatchNormLayers", False),
                                               NoClassificationLayer=Config[
                                                                         'ECAL'] and
                                                                     Config[
                                                                         'HCAL'],
                                               OutputBase=OutputBase)
        ECALModel.Build()
        MyModel = ECALModel

    if HCAL:
        HCALModel = Fully3DImageClassification(Name + "HCAL",
                                               HCALShape,
                                               Config['ECALWidth'],
                                               Config['HCALDepth'],
                                               Config['BatchSize'],
                                               Config['NClasses'],
                                               init=TestDefaultParam(
                                                   "WeightInitialization",
                                                   'random_normal'),
                                               activation=TestDefaultParam(
                                                   "activation", "relu"),
                                               Dropout=TestDefaultParam(
                                                   "DropoutLayers", 0.5),
                                               BatchNormalization=TestDefaultParam(
                                                   "BatchNormLayers", False),
                                               NoClassificationLayer=ECAL and HCAL,
                                               OutputBase=OutputBase)
        HCALModel.Build()
        MyModel = HCALModel

    if HCAL and ECAL:
        MyModel = MergerModel(Name + "_Merged",
                              [ECALModel, HCALModel],
                              Config['NClasses'],
                              Config['WeightInitialization'],
                              OutputBase=OutputBase)

    # Configure the Optimizer, using optimizer configuration parameter.
    MyModel.Loss = Config['loss']
    # Build it
    MyModel.Build()
    print " Done."

if BuildModel:
    print "Output Directory:", MyModel.OutDir
    # Store the Configuration Dictionary
    MyModel.MetaData["Configuration"] = Config
    if "HyperParamSet" in dir():
        MyModel.MetaData["HyperParamSet"] = HyperParamSet

    # Print out the Model Summary
    MyModel.Model.summary()

    # Compile The Model
    print "Compiling Model."
    MyModel.BuildOptimizer(Config['optimizer'], Config)
    MyModel.Compile(Metrics=["accuracy"])

# Train
if Train or (RecoverMode and FailedLoad):
    print "Training."
    # Setup Callbacks
    # These are all optional.


    callbacks = []

    # Still testing this...

    if TestDefaultParam("UseGracefulExit", 0):
        print "Adding GracefulExit Callback."
        callbacks.append(GracefulExit())

    if TestDefaultParam("ModelCheckpoint", False):
        MyModel.MakeOutputDir()
        callbacks.append(
            ModelCheckpoint(MyModel.OutDir + "/Checkpoint.Weights.h5",
                            monitor=TestDefaultParam("monitor", "val_loss"),
                            save_best_only=TestDefaultParam(
                                "ModelCheckpoint_save_best_only"),
                            save_weights_only=TestDefaultParam(
                                "ModelCheckpoint_save_weights_only"),
                            mode=TestDefaultParam("ModelCheckpoint_mode",
                                                  "auto"),
                            period=TestDefaultParam("ModelCheckpoint_period",
                                                    1),
                            verbose=0))

    if TestDefaultParam("EarlyStopping"):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=TestDefaultParam("monitor", "val_loss"),
            min_delta=TestDefaultParam("EarlyStopping_min_delta", 0.01),
            patience=TestDefaultParam("EarlyStopping_patience"),
            mode=TestDefaultParam("EarlyStopping_mode", 'auto'),
            verbose=0))

    if TestDefaultParam("RunningTime"):
        print "Setting Runningtime to", RunningTime, "."
        TSCB = TimeStopping(TestDefaultParam("RunningTime", 3600 * 6),
                            verbose=False)
        callbacks.append(TSCB)

    # Don't fill the log files with progress bar.
#    if sys.flags.interactive:
#        verbose = 1
#    else:
#        verbose = 1  # set to 2

    print "Evaluating score on test sample..."
    tmp = Test_gen.first().generate().next()
    #print('WAT', type(tmp), tmp[0].shape, tmp[1].shape, tmp[2].shape)
    score = MyModel.Model.evaluate_generator(Test_gen.first().generate(),
                                             steps=Config['NTestSamples'] /
                                                   Config['BatchSize'])

    print "Initial Score:", score
    MyModel.MetaData["InitialScore"] = score

    MyModel.History = MyModel.Model.fit_generator(Train_gen.first().generate(),
                                                  steps_per_epoch=(
                                                      NSamples / Config[
                                                          'BatchSize']
                                                  ),
                                                  epochs=Config['Epochs'],
                                                  verbose=verbose,
                                                  validation_data=Test_gen.first().generate(),
                                                  validation_steps=Config[
                                                                       'NTestSamples']
                                                                   / Config[
                                                                       'BatchSize'],
                                                  callbacks=callbacks)

    score = MyModel.Model.evaluate_generator(Test_gen.first().generate(),
                                             steps=Config['NTestSamples'] /
                                                   Config['BatchSize'])

    print "Evaluating score on test sample..."
    print "Final Score:", score
    MyModel.MetaData["FinalScore"] = score

    if TestDefaultParam("RunningTime"):
        MyModel.MetaData["EpochTime"] = TSCB.history

    # Store the parameters used for scanning for easier tables later:
    for k in Params:
        MyModel.MetaData[k] = Config[k]

    # Save Model
    MyModel.Save()
else:
    print "Skipping Training."

# Analysis
# if Analyze:
# print "Running Analysis."

# Test_genC = MakeGenerator(ECAL, HCAL, TestSampleList, Config['NTestSamples,
#                           LCDNormalization(Norms),
#                           batchsize=BatchSize,
#                           shapes=shapes,
#                           n_threads=n_threads,
#                           multiplier=multiplier,
#                           cachefile=Test_genC.cachefilename)

# Test_genC.PreloadData(n_threads_cache)

# TODO what are the following data structures?
# Test_X_ECAL, Test_X_HCAL, Test_Y = tuple(Test_genC.D)

# analysis_gen = H5FileDataProvider(test_sample_spec,
#                                   batch_size=Config['BatchSize'],
#                                   process_function=LCDN(Norms),
#                                   n_readers=2,
#                                   q_multipler=10,
#                                   wrap_examples=True)
#
# result, NewMetaData = MultiClassificationAnalysis(MyModel,
#                                                   [Test_X_ECAL,
#                                                    Test_X_HCAL],
#                                                   Test_Y,
#                                                   Config['BatchSize'],
#                                                   PDFFileName="ROC",
#                                                   IndexMap={0: 'Pi0',
#                                                             2: 'ChPi',
#                                                             3: 'Gamma',
#                                                             1: 'Ele'})
#
# MyModel.MetaData.update(NewMetaData)
#
# # Save again, in case Analysis put anything into the Model MetaData
# if not sys.flags.interactive:
#     MyModel.Save()
# else:
#     print "Warning: Interactive Mode. Use MyModel.Save() to save Analysis Results."

# Make sure all of the Generators processes and threads are dead.
# Not necessary... but ensures a graceful exit.
# if not sys.flags.interactive:
#     for g in GeneratorClasses:
#         try:
#             g.StopFiller()
#             g.StopWorkers()
#         except:
#             pass
