from processDataFiles import *

######################################################################3
# processDataFiles(dataFiles):
######################################################################3
# combines dataFiles (a list of tuples with (fileName, classN))
# returns trainX, testX, trainY, testY

from BDT import *

######################################################################3
# trainBDT(X, y):
######################################################################3
# trains and returns a BDT

######################################################################3
# testBDT(bdt, X):
######################################################################3
# runs a BDT on dataset X 

from analyzeResults import *

######################################################################3
# analyzeResults(model, X, yTruth):
######################################################################3
# compares results of running model on X to yTruth, generating plots such as ROC's
