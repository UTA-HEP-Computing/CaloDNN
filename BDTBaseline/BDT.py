import AITools.AITools as AI

print "Adding files"
DIR = "/data/LCD/"
dataFiles = []
comparison = 0 # 0 for e vs. pi+, 1 for photon vs. pi0

if comparison == 0:
    ELECTRON = 0
    PI_PLUS = 1
    classLabels = ["electron", "pi+"]
    saveDir = "Plots/eVsPi+/"

    for fileN in range(1, 101):
        dataFiles.append((DIR + "e-_60_GeV_normal/e-_60_GeV_normal_" + str(fileN) + ".h5", ELECTRON))
    for fileN in range(20001, 22001):
        dataFiles.append((DIR + "pi+_60_GeV_normal_025_cut/pi+_60_GeV_normal_" + str(fileN) + ".h5", PI_PLUS))
    #for fileN in range(1, 5001):
    #    dataFiles.append((DIR + "pi+_105_GeV_normal_025_cut/pi+_105_GeV_normal_" + str(fileN) + ".h5", PI_PLUS))

elif comparison == 1:
    PHOTON = 0
    PI_ZERO = 1
    classLabels = ["photon", "pi0"]
    saveDir = "Plots/photonVsPi0/"

    for fileN in range(1, 101):
        dataFiles.append((DIR + "gamma_60_GeV_normal/gamma_60_GeV_normal_" + str(fileN) + ".h5", PHOTON))
    for fileN in range(1, 1001):
        dataFiles.append((DIR + "pi0_60_GeV_normal/pi0_60_GeV_normal_" + str(fileN) + ".h5", PI_ZERO))

print "Processing files"
trainX, testX, trainY, testY = AI.processDataFiles(dataFiles)

print "Training BDT"
BDT = AI.trainBDT(trainX, trainY)

print "Analyzing results"
AI.analyzeResults(BDT, testX, testY, classLabels, saveDir)
