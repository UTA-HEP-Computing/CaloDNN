from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# trains and returns a BDT
def trainBDT(X, y):
    dt = DecisionTreeClassifier(max_depth = 5)
    bdt = AdaBoostClassifier(dt,
            algorithm = 'SAMME',
            n_estimators = 800,
            learning_rate = 0.5)
    bdt.fit(X, y)
    return bdt

# runs a BDT on dataset X 
def testBDT(bdt, X):
    return bdt.predict(X)
