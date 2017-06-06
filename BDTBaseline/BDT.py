import AITools.AITools as AI

ELECTRON = 0
PI_PLUS = 1
PHOTON = 2
PI_ZERO = 3

DIR = "../../CaloSampleGeneration/AllFiles/SkimmedH5Files/Wei/"

dataFiles = [
        (DIR + "e-_60_GeV_normal_1000.h5", ELECTRON),
        (DIR + "pi+_105_GeV_normal_1.h5", PI_PLUS)
        ]

trainX, testX, trainY, testY = AI.processDataFiles(dataFiles)

BDT = AI.trainBDT(trainX, trainY)

AI.analyzeResults(BDT, testX, testY)
