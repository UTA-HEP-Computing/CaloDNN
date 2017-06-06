from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# trains and returns a BDT
def trainBDT(X, y,
        max_depth = 5,
        algorithm = 'SAMME',
        n_estimators = 800,
        learning_rate = 0.5
        ):
    dt = DecisionTreeClassifier(max_depth = max_depth)
    bdt = AdaBoostClassifier(dt,
            algorithm = algorithm,
            n_estimators = n_estimators,
            learning_rate = learning_rate,
            )
    bdt.fit(X, y)
    return bdt

# runs a BDT on dataset X 
def testBDT(bdt, X):
    return bdt.predict(X)
