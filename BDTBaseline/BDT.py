import AITools.AITools as AI

ELECTRON = 0
PI_PLUS = 1
PHOTON = 2
PI_ZERO = 3

DIR = "/data/LCD/"
dataFiles = []
print "Adding files"
#for fileN in range(1, 5001):
for fileN in range(1, 101):
    dataFiles.append((DIR + "e-_60_GeV_normal/e-_60_GeV_normal_" + str(fileN) + ".h5", ELECTRON))
#for fileN in range(1, 6000):
for fileN in range(1, 101):
    dataFiles.append((DIR + "pi+_105_GeV_normal/pi+_105_GeV_normal_" + str(fileN) + ".h5", PI_PLUS))

print "Processing files"
trainX, testX, trainY, testY = AI.processDataFiles(dataFiles)
print trainX.shape

print "Training BDT"
BDT = AI.trainBDT(trainX, trainY)

print "Analyzing results"
AI.analyzeResults(BDT, testX, testY)
