import h5py as h5
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

######################################################################3
# processDataFiles(dataFiles):
# combines dataFiles (a list of tuples with (fileName, classN))
# returns trainX, testX, trainY, testY
######################################################################3

# wish there was a more memory-efficient way to do this, but for now we take all features from input data and concat
# into a numpy array
def _appendData(dataFile, classID, X, y):
    features = dataFile.keys()
    featuresToRemove = ["ECAL", "HCAL", "conversion", "energy", "pdgID"]
    for feature in featuresToRemove:
        if feature in features: features.remove(feature)
    for feature in features:
        newFeatureEvents = np.array(dataFile[feature]).astype(np.float)
        if newFeatureEvents.shape[0] > 1: # change column arrays to rows
            newFeatureEvents = newFeatureEvents.flatten()
        if feature in X.keys():
            X[feature] = np.concatenate([X[feature], newFeatureEvents])
        else:
            X[feature] = newFeatureEvents
    nEvents = len(dataFile["ECAL"])
    y += [classID]*nEvents

def _preprocessData(X, y):
    X = np.array(pd.DataFrame(X))
    y = np.array(y)
    y = y[np.isfinite(X).all(axis=1)] # remove events with inf or NaN for any feature
    X = X[np.isfinite(X).all(axis=1)] # have to change X last, because it's used as reference to change y
    return X, y

def _splitSamples(dataX, dataY):
    return train_test_split(dataX, dataY, test_size=0.33, random_state=42)

# combines files (a list of tuples with (fileName, classN)), and returns trainX, testX, trainY, testY
def processDataFiles(dataFiles):
    X = {}
    y = []
    for dataFile in dataFiles:
        fileName = dataFile[0]
        classN = dataFile[1]
        newSample = h5.File(fileName)
        _appendData(newSample, classN, X, y)
    X, y = _preprocessData(X, y)
    return _splitSamples(X, y)
