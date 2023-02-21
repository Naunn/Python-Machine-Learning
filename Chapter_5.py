# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:19:18 2022

@author: Bartosz Lewandowski
"""
# %% Libraries
# Niezbedne
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# %% PCA - unsupervised data compression
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# Normalizacja
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# Macierz kowariancji
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

# the variance explained ratios plot
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp) # Return the cumulative sum of the elements along a given axis.

plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) # Calculate the absolute value element-wise.
               for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# np.hstack
# a = np.array((1,2,3))
# b = np.array((4,5,6))
# np.hstack((a,b))
# array([1, 2, 3, 4, 5, 6])
# a = np.array([[1],[2],[3]])
# b = np.array([[4],[5],[6]])
# np.hstack((a,b))
# array([[1, 4],
#        [2, 5],
#        [3, 6]])

# np.newaxis
# https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis
w= np.hstack((eigen_pairs[0][1][:, np.newaxis],
              eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n',w)

# Przyklad rzutu na nowa podprzestrzen, tj. x'=xW
X_train_std[0].dot(w)
# Rzut wszystkich obserwacji na nowa podprzestrzen
X_train_pca = X_train_std.dot(w)

# Wizualizacja
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
# Caution! PCA is an unsupervised technique that doesn't use class label information.

# %% PCA in scikit-learn
def plot_decision_regions(X, y, classifier,
                          test_idx=None, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # np.meshgrid() - utworzenie 'wspolrzednych' z dwoch list
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    
    # np.ravel() - 'skompresowanie' wszystkich wartosc z list w jedna liste
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(lox='lower left')
plt.show()
# =============================================================================
# The plot above is a mirror image of the previous PCA via our step-by-step
# approach. Note that the reason for this difference is that, depending on
# the eigensolver, eigenvectors can have either negative or positive signs.
# We could simply revert the mirror image by multiplying the data with -1 if we wanted to.
# =============================================================================
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(lox='lower left')
plt.show()

# Trik na sprawdzenie 'poziomu wytlumaczalnosci' przez kolejne zmienne
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_

# %% LDA - supervised data compression 163









