# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:39:27 2022

@author: Bartosz Lewandowski
"""
# %% Libraries
# Niezbedne
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Dane
from sklearn import datasets

# Wizualizacja
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Visualization
def plot_decision_regions(X, y, classifier,
                          test_idx=None, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # Utworzenie listy kolorow z dostepnych powyzej
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Utworzenie osi:
    # arange - Return evenly spaced values within a given interval.
    # meshgrid -
    # xv, yv = np.meshgrid(x, y, indexing='ij')
    #     for i in range(nx):
    #         for j in range(ny):
    #             # treat xv[i,j], yv[i,j]
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # ravel - Return a continuous flattened array.
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    # np.ravel(x)
    # array([1, 2, 3, 4, 5, 6])   
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # contourf - Plot filled contours.
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], facecolors='none',
                    edgecolors='black', alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')
        
# %% Iris
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

np.unique(y) # array([0, 1, 2])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

len(X_train) # 105
len(X_test) # 45

sc = StandardScaler()
# Using the fit method, StandardScaler estimated the parameters µ (sample mean) 
# and σ (standard deviation) for each feature dimension from the training data.
sc.fit(X_train)
# Note that we used the same scaling parameters to standardize the test set so that
# both the values in the training and test dataset are comparable to each other.
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# %% Perceptron (sklearn)
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
# %d, s% - placeholders for integer values, decimals or numbers and strings (respectively)
# https://www.geeksforgeeks.org/difference-between-s-and-d-in-python-string/
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuaracy: %.2f' % round(1-(y_test != y_pred).sum()/len(y_test),2))
print('Accuaracy: %.2f' % accuracy_score(y_test, y_pred))
        
# vstack() - Stack arrays in sequence vertically (row wise).
# a = np.array([[1], [2], [3]])
# b = np.array([[4], [5], [6]])
# np.vstack((a,b))
# array([[1],
#        [2],
#        [3],
#        [4],
#        [5],
#        [6]])
X_combined_std = np.vstack((X_train_std, X_test_std))
# hstack() - Stack arrays in sequence horizontally (column wise).
# a = np.array([[1],[2],[3]])
# b = np.array([[4],[5],[6]])
# np.hstack((a,b))
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# help(Perceptron)

# %% Logistic regression (sklearn)
# Sigmoid
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()

# sklearn
# help(LogisticRegression)
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined, classifier=lr,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

lr.predict_proba(X_test_std[[0],:])
[round(x,4) for x in lr.predict_proba(X_test_std[[0],:])[0]]

# %% L2 regularization
weights, params = [], []
for c in np.arange(-5.0, 5.0):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)

plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

# %% SVM (sklearn)
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# %% SGDClassifier (sklearn) - Stochastic Gradient Descent (for BIG DATA - faster)
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')

# %% SVM for nonlinear problems
# Create nonlienar dataset
np.random.seed(0)
X_xor = np.random.rand(200, 2)
# p q  p xor q
# 0	0	  0
# 0	1	  1
# 1	0	  1
# 1	1	  0
y_xor = np.logical_xor(X_xor[:, 0] > 0.5, X_xor[:, 1] > 0.5)
# help(np.where)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
            c='r', marker='s', label='-1')
# plt.ylim(-3.0)
plt.legend()
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
# =============================================================================
# The γ parameter, which we set to gamma=0.1, can be understood as a cut-off
# parameter for the Gaussian sphere. If we increase the value for γ , we increase the
# influence or reach of the training samples, which leads to a softer decision boundary.
# =============================================================================
svm = SVC(kernel='rbf', random_state=0, gamma=1.10, C=10.0)
svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined, classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined, classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# %% Decision tree
# https://www.youtube.com/watch?v=ZVR2Way4nwQ
# https://www.youtube.com/watch?v=_L39rN6gz7Y&

# visual comparison of the three different impurity criteria
def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
                         ['Entropy', 'Entropy (scaled)',
                          'Gini Impurity',
                          'Misclassification Error'],
                         ['-', '-', '--', '-.'],
                         ['black', 'lightgray',
                          'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab,
                   linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i-1)')
plt.ylabel('Imputiry Index')
plt.show()

# building a decision tree
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

# wizualizacja z uzyciem GraphViz
from sklearn.tree import export_graphviz

export_graphviz(tree,
                out_file='tree.dot',
                feature_names=['petal length', 'petal width'])

# Execute following in cmd (in saved file location)
# dot -Tpng tree.dot -o tree.png

# %% Random forest (sklearn)
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
# n_jobs - allows us to parallelize the model training using multiple cores
# of our computer, here, two
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()

# %% KNN (sklearn)
knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')
# The 'minkowski' distance is a generalization of the Euclidean and
# Manhattan distance It becomes the Euclidean distance if we set the
# parameter p=2 or the Manhatten distance at p=1, respectively.
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()