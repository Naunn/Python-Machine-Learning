# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:17:02 2022

@author: Bartosz Lewandowski
"""
# %% Libraries
# Niezbedne
import pandas as pd
import numpy as np
from numpy.random import seed

# Wizualizacja
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# %% Perceptron (MCP)
# =============================================================================
# The whole idea behind the MCP neuron and Rosenblatt's thresholded perceptron 
# model is to use a reductionist approach to mimic how a single neuron in
# the brain works: it either fires or it doesn't. Thus, Rosenblatt's initial
# perceptron rule is fairly simple and can be summarized by the following steps:
#   1. Initialize the weights to 0 or small random numbers.
#   2. For each training sample x^(i) perform the following steps:
#       1. Compute the output value \hat{y}.
#       2. Update the weights.
# =============================================================================
class Perceptron:
    """Perceptron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
        
    Attribiutes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of missclassifications in every epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : object
        
        """
        # Wektor wag (alterantywnie moga byc losowe, niewielkie, wagi)
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Rownanie na delte (wagi)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # Aktualizacja pierwszej wagi (thety)
                self.w_[0] += update
                # Sledzenie ilosci poprawnie/negatywnie zaklasyfikowanych danych
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
# %% Adaptive Linear Neuron - ADALINE (Batch Gradient Descent)
# =============================================================================
# We minimize a cost function by taking a step into the opposite direction
# of a gradient that is calculated from the whole training set; this is why
# this approach is sometimes also referred to as batch gradient descent.
# =============================================================================
class AdalineGD:
    """"ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
        
    Attribiutes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of missclassifications in every epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            # "wnetrze" funkcji kosztu (w tym przypadku (SSE))
            errors = (y - output)
            # Aktualizacja wag zgodnie z pochodna funkcji kosztu
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # Funkcja kosztu (SSE)
            cost = (errors**2).sum() / 2.0
            # Sledzenie wartosci funkcji kosztu
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# %% Stochastic Gradient Descent (SGD)
# =============================================================================
# A popular alternative to the batch gradient descent algorithm is stochastic
# gradient descent, sometimes also called iterative or on-line gradient descent.
# Instead of updating the weights based on the sum of the accumulated errors over
# all samples x^(i), we update the weights incrementally for each training sample.
# https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent
# =============================================================================
class AdalineSGD:
    """"ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
        
    Attribiutes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of missclassifications in every epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch
        if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling
        and initializing the weights.
    
    """
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : object
        
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        # Dla kazdej iteracji
        for i in range(self.n_iter):
            if self.shuffle:
                # "przetasowanie" danych
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                # Obliczenie wartosci funkcji kosztu
                # (zdefiniowane w oddzielnej metodzie)
                cost.append(self._update_weights(xi, target))
            # Sprawdzenie zachowania sredniego kosztu (czy rosnie, czy maleje)
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)          
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        # Trenowanie modelu (dopasowanie wag) dla nowych danych,
        # bez zmiany poprzednio wytrenowanych wag
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """Shuffle training data"""
        # "Przetasowuje" wektor, a nastepnie wybiera dane wedlug tego wektora
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        # Wektor wag o okreslonej dlugosci rownej m
        # (alterantywnie moga byc losowe, niewielkie, wagi)
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
        
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
# %% data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail(10)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

# %% training
# Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
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
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.show()

# AdalineGD
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
           np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
           ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# AdalineGD - training with standarization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1),
           ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

# AdalineSGD
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Avearge Cost')
plt.show()