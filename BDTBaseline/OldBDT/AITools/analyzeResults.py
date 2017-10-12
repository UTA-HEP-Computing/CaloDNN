import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import os

# compares y to yTruth, generating plots such as ROC's
def analyzeResults(model, X, yTruth, classLabels, saveDir):

    ###############
    # Preparation #
    ###############

    nClasses = len(classLabels)
    if not os.path.exists(saveDir): os.makedirs(saveDir)

    #############
    # Run model #
    #############

    if isinstance(model, AdaBoostClassifier):
        scores = model.decision_function(X)
        y = model.predict(X)
    else:
        print "Unknown model type:", type(model)

    #########################
    # Make tables and plots #
    #########################

    def plotROC(falsePosRate, truePosRate, thresholds, saveName):
        plt.plot(falsePosRate, truePosRate, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(saveName, bbox_inches="tight")
        plt.clf()

    print (classification_report(yTruth, y, target_names=classLabels))

    if nClasses == 2:
        roc_auc = roc_auc_score(yTruth, scores)
        falsePosRate, truePosRate, thresholds = roc_curve(yTruth, scores) 
        print ("Area under ROC curve: %.4f" % roc_auc)
        plotROC(falsePosRate, truePosRate, thresholds, saveName=saveDir+"ROC.png")
    else:
        for classN in range(nClasses): 
            classYTruth = [y==classN for y in yTruth]
            classScores = scores[:,classN]
            classLabel = classLabels[classN]
            roc_auc = roc_auc_score(classYTruth, classScores)
            falsePosRate, truePosRate, thresholds = roc_curve(classYTruth, classScores) 
            print ("Area under ROC curve for " + str(classLabel) + ": %.4f" % roc_auc)
            plotROC(falsePosRate, truePosRate, thresholds, saveName=saveDir+str(classLabel)+"_ROC.png")
