from processDataFiles import *

######################################################################
# processDataFiles(dataFiles):
# combines dataFiles (a list of tuples with (fileName, classN))
# returns trainX, testX, trainY, testY
######################################################################

from BDT import *

######################################################################
# trainBDT(X, y):
# trains and returns a BDT
#---------------------------------------------------------------------
# max_depth = 5
# algorithm = 'SAMME'
# n_estimators = 800
# learning_rate = 0.5
######################################################################

######################################################################
# testBDT(bdt, X):
# runs a BDT on dataset X 
######################################################################

from analyzeResults import *

######################################################################
# analyzeResults(model, X, yTruth, classLabels, saveDir):
# compares results of running model on X to yTruth, generating plots such as ROC's
######################################################################
