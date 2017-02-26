# Takes min max arguments and runs N jobs as nohup
# Backup when you don't have a queue

import sys,subprocess

if len(sys.argv)<2:
    print "Error. Must specify min and max hyperparameter set numbers."
    exit()
    
minS=int(sys.argv[1])
maxS=int(sys.argv[2])

iGPU=0
for s in xrange(minS,maxS):
    command = "nohup python -m CaloDNN.ClassificationExperiment -s "+str(0)+" --gpu "+str(iGPU)+" >> ScanLogs/"+str(s)+".log & "
    iGPU+=1
    print command
    subprocess.Popen(command,shell=True)

    
