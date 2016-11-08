# Analysis
import numpy as np

def ClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize, SignalClassIndex=5):
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt

    from sklearn.metrics import roc_curve, auc    

    print "Prediction Analysis."
    result = MyModel.Model.predict(Test_X, batch_size=BatchSize)
    
    fpr, tpr, _ = roc_curve(Test_Y[:,SignalClassIndex], 
                            result[:,SignalClassIndex])
    roc_auc = auc(fpr, tpr)    
    
    lw=2

    plt.plot(fpr,tpr,color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    print "ROC AUC: ",roc_auc

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")

    plt.savefig(MyModel.OutDir+"/ROC.pdf")


mpColors=["blue","green","red","cyan","magenta","yellow","black","white"]

def MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize):
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt

    from sklearn.metrics import roc_curve, auc    

    print "Prediction Analysis."
    result = MyModel.Model.predict(Test_X, batch_size=BatchSize)
    
    NClasses=Test_Y.shape[1]

    for ClassIndex in xrange(0,NClasses):
        fpr, tpr, _ = roc_curve(Test_Y[:,ClassIndex], 
                                result[:,ClassIndex])
        roc_auc = auc(fpr, tpr)    
    
        lw=2

        plt.plot(fpr,tpr,color=mpColors[ClassIndex],
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

        print "ROC ",ClassIndex," AUC: ",roc_auc

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc="lower right")
        

    plt.savefig(MyModel.OutDir+"/ROC.pdf")

    return result
