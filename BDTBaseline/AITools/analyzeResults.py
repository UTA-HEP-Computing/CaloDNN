from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# compares y to yTruth, generating plots such as ROC's
def analyzeResults(model, X, yTruth):

    #############
    # Run model #
    #############

    if isinstance(model, AdaBoostClassifier):
        scores = model.decision_function(X)
        y = model.predict(X)
    else:
        print "Unknown model type:", type(model)

    #################
    # Print reports #
    #################

    print (classification_report(yTruth, y))

    roc_auc = roc_auc_score(yTruth, scores)
    print ("Area under ROC curve: %.4f" % roc_auc)

    ##############
    # Make plots #
    ##############

    falsePosRate, truePosRate, thresholds = roc_curve(yTruth, scores) 

    # ROC
    plt.plot(falsePosRate, truePosRate, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
