import graphviz
import os
import csv
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

def csvrow2arr(row):
    arr = [ float(w.strip()) for w in row.split(',') ]
    return arr


def load_data(path = './data/resnet_cifar10', type='mean', train=True):  # type = 'mean' or 'median' or 'both'
    file_path = '%s/train.csv' % path
    if not train :
        file_path = '%s/test.csv' % path

    assert os.path.exists(path)
    assert os.path.isfile(file_path)

    means = []
    medians = []

    if type == 'mean' or type == 'both':
        mean_path = '%s/mean.csv' % path
        with open(mean_path, 'r') as fin:
            means = csvrow2arr( fin.readline() )

    if type == 'median' or type == 'both':
        median_path = '%s/median.csv' % path
        with open(median_path, 'r') as fin:
            medians = csvrow2arr( fin.readline() )

    X = []
    Y = []

    with open(file_path, 'r') as fin:
        for line in fin.readlines():

            data = csvrow2arr(line)

            xdata = data[1:]
            xvalue = []
            yvalue = int(data[0])

            for i in range(len(means)):
                if xdata[i] <= means[i]:
                    xvalue.append(0)
                else:
                    xvalue.append(1)

            for i in range(len(medians)):
                if xdata[i] <= medians[i]:
                    xvalue.append(0)
                else:
                    xvalue.append(1)

            X.append(xvalue)
            Y.append(yvalue)

    return X, Y


X_train, Y_train = load_data(type='mean', train=True)
X_test, Y_test = load_data(type='mean', train=False)

tree = RandomForestClassifier(max_depth=5)
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