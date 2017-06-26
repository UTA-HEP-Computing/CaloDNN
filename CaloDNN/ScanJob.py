# Takes min max arguments and runs N jobs as nohup
# Backup when you don't have a queue

import subprocess
import sys

if len(sys.argv) < 2:
    print "Error. Must specify min and max hyperparameter set numbers."
    exit()

minS = int(sys.argv[1])
maxS = int(sys.argv[2])

Delay = 1800
iGPU = 0

if len(sys.argv) > 3:
    iGPU = int(sys.argv[3])

TestMode = False

for s in xrange(minS, maxS):
    if TestMode:
        # Submit exactly same job every 30 mins... use for scaling study
        command = "sleep " + str(
            iGPU * Delay) + " ; nohup python -m CaloDNN.ClassificationExperiment -s " + str(
            0) + " --gpu " + str(iGPU) + " >> ScanLogs/" + str(s) + ".log & "
    else:
        command = "nohup python -m CaloDNN.ClassificationExperiment -s " + str(
            s) + " --gpu " + str(iGPU) + " >> ScanLogs/" + str(s) + ".log & "
    iGPU += 1
    print command
    subprocess.Popen(command, shell=True)
