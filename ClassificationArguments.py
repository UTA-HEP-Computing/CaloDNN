# Configuration of this jobConfig
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',default="CaloDNN/ClassificationScanConfig.py", help="Use specified configuration file.")

parser.add_argument('-L', '--LoadModel',help='Loads a model from specified directory.', default=False)
parser.add_argument('--gpu', dest='gpuid', default="", help='Use specified GPU.')
parser.add_argument('--cpu', action="store_true", help='Use CPU.')
parser.add_argument('--NoTrain', action="store_true", help="Do not run training.")
parser.add_argument('--NoAnalysis', action="store_true", help="Do not run analysis.")
parser.add_argument('--LowMem', action="store_true", help="Minimize Memory Usage.")
parser.add_argument('--Test', action="store_true", help="Run in test mode (reduced examples and epochs).")
parser.add_argument('--Recover', action="store_true", help="Train only if fail to load model (use with --NoTrain and --Load or --LoadPrevious).")
parser.add_argument('-s',"--hyperparamset", default="0", help="Use specificed (by index) hyperparameter set.")
parser.add_argument('--nopremix', action="store_true", help="Do not use the premixed inputfile. Mix on the fly.")
parser.add_argument('--preload', action="store_true", help="Preload the data into memory. Caution: requires lots of memory.")
parser.add_argument('-r',"--runningtime", default="0", help="End training after specified number of seconds.")
parser.add_argument('-p',"--LoadPrevious", action="store_true", help="Load the last trained model.")
parser.add_argument('--GracefulExit', action="store_true", help="Enable graceful exit via Ctrl-C or SIGTERM signal.")

#parser.add_argument('--generator', action="store_true")

# Configure based on commandline flags... this really needs to be cleaned up
args = parser.parse_args()
Train = not args.NoTrain
Analyze = not args.NoAnalysis
TestMode = args.Test
RecoverMode = args.Recover
UseGPU = not args.cpu
gpuid = args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)
ConfigFile = args.config
#useGenerator = args.generator
Premix = not args.nopremix
Preload= args.preload
LoadPreviousModel=args.LoadPrevious
LoadModel=args.LoadModel
LowMemMode=args.LowMem
UseGracefulExit=args.GracefulExit

if int(args.runningtime)>0:
    RunningTime=int(args.runningtime)

# Configuration from PBS:
if "PBS_ARRAYID" in os.environ:
    HyperParamSet = int(os.environ["PBS_ARRAYID"])

if "PBS_QUEUE" in os.environ:
    if "cpu" in os.environ["PBS_QUEUE"]:
        UseGPU=False
    if "gpu" in os.environ["PBS_QUEUE"]:
        UseGPU=True
        gpuid=int(os.environ["PBS_QUEUE"][3:4])

if UseGPU:
    print "Using GPU",gpuid
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%s,floatX=float32,force_device=True" % (gpuid)
else:
    print "Using CPU."

