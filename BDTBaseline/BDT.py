import AITools.AITools as AI

ELECTRON = 0
PI_PLUS = 1
PHOTON = 2
PI_ZERO = 3

classLabels = ["electron", "pi+", "photon", "pi0"]
saveDir = "Plots/4-class/"

DIR = "/data/LCD/"
dataFiles = []
print "Adding files"
for fileN in range(1, 6):
    dataFiles.append((DIR + "e-_60_GeV_normal/e-_60_GeV_normal_" + str(fileN) + ".h5", ELECTRON))
for fileN in range(1, 251):
    dataFiles.append((DIR + "pi+_105_GeV_normal/pi+_105_GeV_normal_" + str(fileN) + ".h5", PI_PLUS))
for fileN in range(1, 6):
    dataFiles.append((DIR + "gamma_60_GeV_normal/gamma_60_GeV_normal_" + str(fileN) + ".h5", PHOTON))
for fileN in range(1, 51):
    dataFiles.append((DIR + "pi0_60_GeV_normal/pi0_60_GeV_normal_" + str(fileN) + ".h5", PI_ZERO))

print "Processing files"
trainX, testX, trainY, testY = AI.processDataFiles(dataFiles)
print trainX.shape

print "Training BDT"
BDT = AI.trainBDT(trainX, trainY)

print "Analyzing results"
AI.analyzeResults(BDT, testX, testY, classLabels, saveDir)
