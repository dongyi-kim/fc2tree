import graphviz
import os
import csv
import numpy as np 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

def csvrow2arr(row):
    arr = [ w.strip() for w in row.split(',') ]
    return arr


def load_data(path = './data/resnet_cifar100', type='mean', K=3):  # type = 'mean' or 'median' or 'both'
    train_file_path = '%s/train.csv' % path
    test_file_path  = '%s/test.csv' % path

    assert os.path.exists(path)
    assert os.path.isfile(train_file_path)
    assert os.path.isfile(test_file_path)
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    with open(train_file_path, 'r') as fin:
        for line in fin.readlines():
            raw_data = csvrow2arr(line)

            xdata = [ float(x) for x in raw_data[1:] ] 
            ydata = int(raw_data[0])
  
            X_train.append(xdata)
            Y_train.append(ydata)

    with open(test_file_path, 'r') as fin:
        for line in fin.readlines():
            raw_data = csvrow2arr(line)

            xdata = [ float(x) for x in raw_data[1:] ] 
            ydata = int(raw_data[0])
  
            X_test.append(xdata)
            Y_test.append(ydata)

    nTrainData = len(X_train)
    nTestData = len(X_test)
    nFeatures = len(X_train[0])
    nClasses  = max( np.max(Y_train), np.max(Y_test) )

    if type == 'mean':
        X_mean = np.mean(X_train, axis=0)
        for i in range(nTrainData):
            for j in range(nFeatures):
                X_train[i][j] = 0 if X_train[i][j] <= X_mean[j] else 1

        for i in range(nTestData):
            for j in range(nFeatures):
                X_test[i][j] = 0 if X_test[i][j] <= X_mean[j] else 1
                
    elif type == 'median':
        X_median = np.median(X_train, axis=0)
        for i in range(nTrainData):
            for j in range(nFeatures):
                X_train[i][j] = 0 if X_train[i][j] <= X_median[j] else 1

        for i in range(nTestData):
            for j in range(nFeatures):
                X_test[i][j] = 0 if X_test[i][j] <= X_median[j] else 1
                
    elif type == 'kmeans':
        new_X_train = [ [] for i in range(nTrainData)] 
        new_X_test =  [ [] for i in range(nTestData) ]

        for d in range(nFeatures):
            X_data = [ [ X_train[i][d] ] for i in range(nTrainData) ]
            kmeans = KMeans(n_clusters=K)
            kmeans.fit(X_data)

            for i in range(nTrainData):
                idx =  kmeans.predict( [  [X_train[i][d]] ] )[0]
                vec = [0] * K
                vec[idx] = 1
                new_X_train[i] += vec 
            
            for i in range(nTestData):
                idx =  kmeans.predict( [ [ X_test[i][d] ] ] )[0]
                vec = [0] * K
                vec[idx] = 1
                new_X_test[i] += vec 

            print(d)
            print(new_X_train[0])

        X_train = new_X_train
        X_test = new_X_test
        
    return  X_train, Y_train, X_test, Y_test 

X_train, Y_train, X_test, Y_test = load_data(type='mean', K=10)

tree = RandomForestClassifier(n_estimators=5 , max_depth=5, min_samples_split=50)
# tree = DecisionTreeClassifier(random_state=2, max_depth=5)
tree.fit(X_train, Y_train)

# print tree.n_features_
# print tree.feature_importances_

print('training accuracy: {:3f}'.format(tree.score(X_train,Y_train)))
print('testing accuracy : {:3f}'.format(tree.score(X_test,Y_test)))

# export_graphviz(tree, out_file='tree.dot', class_names=[ str(i) for i in range(10) ], feature_names=[ str(i) for i in range(64)], impurity=False, filled=True)
#
# with open('tree.dot') as f:
#     dot_graph = f.read()
#     dot = graphviz.Source(dot_graph)
#     dot.format='png'
#     dot.render(filename='tree.png')