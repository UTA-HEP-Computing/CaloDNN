# Analysis
import numpy as np
from ROOT import TH1F,TCanvas,TF1

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

def MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize,IndexMap=False):
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

        if IndexMap:
            ClassName=IndexMap[ClassIndex]
        else:
            ClassName="Class "+str(ClassIndex)
        
        plt.plot(fpr,tpr,color=mpColors[ClassIndex],
                 lw=lw, label=ClassName+ ' (area = %0.2f)' % roc_auc)

        print "ROC ",ClassIndex," AUC: ",roc_auc

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc="lower right")
        

    plt.savefig(MyModel.OutDir+"/ROC.pdf")

    return result


# From Regression Analysis

def RegressionAnalysis(ModelH,X_test,y_test,M_min,M_max,BatchSize):

    print "Perdiction Analysis."
    result = ModelH.Model.predict(X_test, batch_size=BatchSize)
    MassNorm=(M_max-M_min)

    result=result*MassNorm-M_min

    c1=TCanvas("c1")

    resultHist=TH1F("Result","Result",100,1.1*M_min,1.1*M_max)
    map(lambda(x): resultHist.Fill(x), result.flatten())
    resultHist.Draw()
    c1.Print(ModelH.OutDir+"/Result.pdf")

    targetHist=TH1F("Target","Target",100,1.1*M_min,1.1*M_max)
    map(lambda(x): targetHist.Fill(x), y_test)
    targetHist.Draw()
    c1.Print(ModelH.OutDir+"/Target.pdf")

    residual=result.flatten()-y_test

    residualHist=TH1F("Residual","Residual",100,-5,5)
    map(lambda(x): residualHist.Fill(x), residual)
    # residualHist.Fit("gaus")

    # fit= residualHist.GetFunction("gaus")
    # chi2 = fit.GetChisquare()
    # A = fit.GetParameter(0)
    # A_sigma = fit.GetParError(0)

    # mean = fit.GetParameter(1)
    # mean_sigma = fit.GetParError(1)
    
    # sigma = fit.GetParameter(2)
    # sigma_sigma = fit.GetParError(2)

    # ModelH.MetaData["ResidualMean"]=residualHist.GetMean()
    # ModelH.MetaData["ResidualStdDev"]=residualHist.GetStdDev()

    # ModelH.MetaData["ResidualFitChi2"]=chi2
    # ModelH.MetaData["ResidualFitMean"]=[mean,mean_sigma]
    # ModelH.MetaData["ResidualFitSigma"]=[sigma,sigma_sigma]

    residualHist.Draw()
    c1.Print(ModelH.OutDir+"/Residual.pdf")
    
    G1 = TF1 ("G1","gaus",-10.0,10.)
    G2 = TF1 ("G2","gaus",-10.,10.0)

    # residualHist.Fit(G1,"R","",-10.,1.)
    # residualHist.Fit(G2,"R","",-1,10.)
     
    # DoubleG = TF1 ("DoubleG","gaus(0)+gaus(3)",-10.0,10.0);
    
    # DoubleG.SetParameter(0,G1.GetParameter(0))
    # DoubleG.SetParameter(1,G1.GetParameter(1))
    # DoubleG.SetParameter(2,G1.GetParameter(2))
    
    # DoubleG.SetParameter(3,G2.GetParameter(0))
    # DoubleG.SetParameter(4,G2.GetParameter(1))
    # DoubleG.SetParameter(5,G2.GetParameter(2))

    # residualHist.Fit(DoubleG)

    fitres=[]

    for ii in xrange(0,6):
        fitres+=[[DoubleG.GetParameter(ii), DoubleG.GetParError(ii)]]

    ModelH.MetaData["DoubleGaussianFit"]=fitres

    residualHist.Draw()
    c1.Print(ModelH.OutDir+"/Residual_2GFit.pdf")

def ClassificationAnalysis(ModelH,X_test,y_test,y_testT,M_min,M_max,NBins,BatchSize):
    print "Perdiction Analysis."

    resultClass = ModelH.Model.predict(X_test, batch_size=BatchSize)

    binwidth=(M_max-M_min)/NBins
    result=np.argmax(resultClass,axis=1)*binwidth+M_min

    c1=TCanvas("c1")

    resultHist=TH1F("Result","Result",NBins,M_min,M_max)
    map(lambda(x): resultHist.Fill(x), result.flatten())
    resultHist.Draw()
    c1.Print(ModelH.OutDir+"/Result.pdf")

    targetHist=TH1F("Target","Target",NBins,M_min,M_max)
    map(lambda(x): targetHist.Fill(x), y_testT)
    targetHist.Draw()
    c1.Print(ModelH.OutDir+"/Target.pdf")

    residual=result.flatten()-y_testT

    residualHist=TH1F("Residual","Residual",NBins,M_min,M_max)
    map(lambda(x): residualHist.Fill(x), residual)
    # residualHist.Fit("gaus")

    # fit= residualHist.GetFunction("gaus")
    # chi2 = fit.GetChisquare()
    # A = fit.GetParameter(0)
    # A_sigma = fit.GetParError(0)

    # mean = fit.GetParameter(1)
    # mean_sigma = fit.GetParError(1)
    
    # sigma = fit.GetParameter(2)
    # sigma_sigma = fit.GetParError(2)

    # ModelH.MetaData["ResidualMean"]=residualHist.GetMean()
    # ModelH.MetaData["ResidualStdDev"]=residualHist.GetStdDev()

    # ModelH.MetaData["ResidualFitChi2"]=chi2
    # ModelH.MetaData["ResidualFitMean"]=[mean,mean_sigma]
    # ModelH.MetaData["ResidualFitSigma"]=[sigma,sigma_sigma]

    residualHist.Draw()
    c1.Print(ModelH.OutDir+"/Residual.pdf")
    
    # G1 = TF1 ("G1","gaus",-10.0,10.)
    # G2 = TF1 ("G2","gaus",-10.,10.0)

    # residualHist.Fit(G1,"R","",-10.,1.)
    # residualHist.Fit(G2,"R","",-1,10.)
     
    # DoubleG = TF1 ("DoubleG","gaus(0)+gaus(3)",-10.0,10.0);
    
    # DoubleG.SetParameter(0,G1.GetParameter(0))
    # DoubleG.SetParameter(1,G1.GetParameter(1))
    # DoubleG.SetParameter(2,G1.GetParameter(2))
    
    # DoubleG.SetParameter(3,G2.GetParameter(0))
    # DoubleG.SetParameter(4,G2.GetParameter(1))
    # DoubleG.SetParameter(5,G2.GetParameter(2))

    # residualHist.Fit(DoubleG)

    # fitres=[]

    # for ii in xrange(0,6):
    #     fitres+=[[DoubleG.GetParameter(ii), DoubleG.GetParError(ii)]]

    # ModelH.MetaData["DoubleGaussianFit"]=fitres

    residualHist.Draw()
    c1.Print(ModelH.OutDir+"/Residual_2GFit.pdf")
    return result,resultClass

