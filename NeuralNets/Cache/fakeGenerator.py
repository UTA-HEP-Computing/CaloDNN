import os
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def _appendData(dataFile, classID, X_ECAL, X_HCAL, y):
    ECAL = np.array(dataFile["ECAL"]).astype(np.float)
    HCAL = np.array(dataFile["HCAL"]).astype(np.float)
    X_ECAL.append(ECAL)
    X_HCAL.append(HCAL)
    nEvents = len(ECAL)
    if classID == 0:
	y += [[1, 0]]*nEvents
    elif classID == 1:
	y += [[0, 1]]*nEvents

def _splitSamples(dataX, dataY):
    return train_test_split(dataX, dataY, test_size=0.33, random_state=42)

if __name__ == "__main__":

    gammaPath="/data/LCD/V2/MLDataset/GammaEscan/"
    pi0Path="/data/LCD/V2/MLDataset/Pi0Escan/"

    GAMMA = 0
    PI0 = 1
    classLabels = ["photon", "pi0"]

    dataFiles = []
    for fileN in range(1):
	dataFiles.append((gammaPath + "GammaEscan_" + str(fileN) + ".h5", GAMMA))
	dataFiles.append((pi0Path + "Pi0Escan_" + str(fileN) + ".h5", PI0))

    X_ECAL = []
    X_HCAL = []
    y = []
    for fileN, dataFile in enumerate(dataFiles):
        fileName = dataFile[0]
        classN = dataFile[1]
        print "Processing file", fileN+1, "out of", len(dataFiles)
        if os.path.isfile(fileName):
            newSample = h5.File(fileName)
            _appendData(newSample, classN, X_ECAL, X_HCAL, y)
        else:
            print fileName, "is not a valid file."

    X_ECAL = np.vstack(X_ECAL)
    X_HCAL = np.vstack(X_HCAL)

    X_indices = range(len(X_ECAL))
    trainX, testX, trainY, testY = _splitSamples(X_indices, y)

    test = h5.File('Test.h5','w')
    test.create_dataset('dset0', data=X_ECAL[testX])
    test.create_dataset('dset1', data=X_HCAL[testX])
    test.create_dataset('dset2', data=testY)
    test.close()

    train = h5.File('Train.h5','w')
    train.create_dataset('dset0', data=X_ECAL[trainX])
    train.create_dataset('dset1', data=X_HCAL[trainX])
    train.create_dataset('dset2', data=trainY)
    train.close()
