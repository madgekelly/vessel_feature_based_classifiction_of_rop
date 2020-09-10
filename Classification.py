from sklearn import tree


def get_statistics(prediction, true):
    TN = len(prediction[(prediction==0) & (true==0)])
    TP = len(prediction[(prediction==1) & (true==1)])
    FN = len(prediction[(prediction==0) & (true==1)])
    FP = len(prediction[(prediction==1) & (true==0)])
    spec = TN/(TN + FP)
    sen = TP/(TP + FN)
    acc = (TP + TN)/(TP + TN + FN + FP)
    return sen, spec, acc


def train_tree(train_x, train_y):
    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
    clf = clf.fit(train_x, train_y)
    return clf


def predict(clf, test_x):
    predictions = clf.predict(test_x)
    return predictions