# Configuration of this job
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',default="CaloDNN/ClassificationScanConfig.py")

parser.add_argument('-L', '--LoadModel',default=False)
parser.add_argument('--gpu', dest='gpuid', default="")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--NoTrain', action="store_true")
parser.add_argument('--NoAnalysis', action="store_true")
parser.add_argument('--Test', action="store_true")
parser.add_argument('-s',"--hyperparamset", default="0")
parser.add_argument('--generator', action="store_true")

# Configure based on commandline flags... this really needs to be cleaned up
args = parser.parse_args()
Train = not args.NoTrain
Analyze = not args.NoAnalysis
TestMode = not args.Test
UseGPU = not args.cpu
gpuid = args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)
ConfigFile = args.config
useGenerator = args.generator

LoadModel=args.LoadModel

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

