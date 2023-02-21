# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:45:47 2022

@author: Bartosz Lewandowski
"""
# %% Libraries
# Niezbedne
import pandas as pd
import numpy as np

from io import StringIO

# %% Missing values
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

# Sprawdzenie pustych wartosci
df.isnull()
#        A      B      C      D
# 0  False  False  False  False
# 1  False  False   True  False
# 2  False  False  False   True
df.isnull().sum()
# A    0
# B    0
# C    1
# D    1
# dtype: int64

# Trik na dane do sckikit-learn'a
df.values
# array([[ 1.,  2.,  3.,  4.],
#        [ 5.,  6., nan,  8.],
#        [ 10., 11., 12., nan]])

# Usun wiersze/kolumny z brakujacymi wartosciami
df.dropna()
#      A    B    C    D
# 0  1.0  2.0  3.0  4.0
df.dropna(axis=1)
#      A     B
# 0  1.0   2.0
# 1  5.0   6.0
# 2  0.0  11.0

# Reszta mozliwosci dropna():
df.dropna(how='all') # only drop rows where all columns are NaN
#       A     B     C    D
# 0   1.0   2.0   3.0  4.0
# 1   5.0   6.0   NaN  8.0
# 2  10.0  11.0  12.0  NaN
df.dropna(thresh=4) # drop rows that have not at least 4 non-NaN values
#      A    B    C    D
# 0  1.0  2.0  3.0  4.0
df.dropna(subset=['C']) # only drop rows where NaN appear in specific columns (here: 'C')
#       A     B     C    D
# 0   1.0   2.0   3.0  4.0
# 2  10.0  11.0  12.0  NaN

# %% Interpolation techniques

# Numerical
# New in version 0.20: SimpleImputer replaces the previous sklearn.preprocessing.
# Imputer estimator which is now removed.
from sklearn.impute import SimpleImputer

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)
# [[ 1.   2.   3.   4. ]
#  [ 5.   6.   7.5  8. ]
#  [10.  11.  12.   6. ]]

# Categorical (nominal - red/blue, ordinal - L>M>S, w kontekscie koszulek)
df = pd.DataFrame([
    ['green', 'M', 10.1,'class1'],
    ['red', 'L', 13.5,'class2'],
    ['blue', 'XL', 15.3,'class1']])
df.columns = ['color','size','price','classlabel']
print(df)
#    color size  price classlabel
# 0  green    M   10.1     class1
# 1    red    L   13.5     class2
# 2   blue   XL   15.3     class1

# Mapowanie wartosci porzadkowych
# (przy zalozeniu, ze XL=L+1=M+2)
size_mapping = {
    'XL':3,
    'L':2,
    'M':1}
df['size'] = df['size'].map(size_mapping)
print(df)
#    color  size  price classlabel
# 0  green     1   10.1     class1
# 1    red     2   13.5     class2
# 2   blue     3   15.3     class1
inv_size_mapping = {v: k for k,v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
#    color size  price classlabel
# 0  green    M   10.1     class1
# 1    red    L   13.5     class2
# 2   blue   XL   15.3     class1

# Encoding class labels (nominalne)
# We need to remember that class labels are not ordinal, and it doesn't matter
# which integer number we assign to a particular string-label.
class_mapping = {label:idx for idx,label in
                 enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
#    color size  price  classlabel
# 0  green    M   10.1           0
# 1    red    L   13.5           1
# 2   blue   XL   15.3           0
inv_class_mapping = {v: k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
#    color size  price classlabel
# 0  green    M   10.1     class1
# 1    red    L   13.5     class2
# 2   blue   XL   15.3     class1

from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
# [0 1 0]
class_le.inverse_transform(y)
# array(['class1', 'class2', 'class1'], dtype=object)

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:,0])
print(X)
# [[1 'M' 10.1]
#  [2 'L' 13.5]
#  [0 'XL' 15.3]]

# One-hot encoding (for unordered data - nominal)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df[['color']])

# Ordinal encoding (for ordered data - ordinal)
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
oe.fit_transform(df[['size']])

# LabelEncoder for labels!

# Get_dummies method
pd.get_dummies(df[['color', 'size', 'price']])

# %% Partitioning a dataset
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
df_wine.head()
print('Class labels', np.unique(df_wine['Class label']))

# random partitioning
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

# %% Feature scaling
# Normalization (=/= normalizacja - w tym przypdaku mowa o uspojnienie skal miedzy kolumnami)
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Standarization (centrowanie - zmiana wartosci na rozklad N(0, 1))
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# =============================================================================
# We fit the StandardScaler only once on the training data and use those
# parameters to transform the test set or any new data point.
# =============================================================================

# %% Selecting meaningful features (to prevent overfitting)
# =============================================================================
# A reason for overfitting is that our model is too complex for the given training
# data and common solutions to reduce the generalization error are listed as follows:
# • Collect more training data.
# • Introduce a penalty for complexity via regularization.
# • Choose a simpler model with fewer parameters.
# • Reduce the dimensionality of the data.
# =============================================================================
from sklearn.linear_model import LogisticRegression

# L2 regularization ("thick" - slabe zerowanie sie wag)
lr = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')
lr.fit(X_train_std, y_train)

print('Training accuaracy:', lr.score(X_train_std, y_train))
print('Test accuaracy:', lr.score(X_test_std, y_test))

lr.intercept_ # array([-0.64732881, -0.43279492, -0.83258577])
lr.coef_
# array([[ 0.58228361,  0.04305595,  0.27096654, -0.53333363,  0.00321707,
#          0.29820868,  0.48418851, -0.14789735, -0.00451997,  0.15005795,
#          0.08295104,  0.38799131,  0.80127898],
#        [-0.71490217, -0.35035394, -0.44630613,  0.32199115, -0.10948893,
#         -0.03572165,  0.07174958,  0.04406273,  0.20581481, -0.71624265,
#          0.39941835,  0.17538899, -0.72445229],
#        [ 0.18373457,  0.32514838,  0.16359432,  0.15802432,  0.09025052,
#         -0.20530058, -0.53304855,  0.1117135 , -0.21005439,  0.62841547,
#         -0.4911972 , -0.55819761, -0.04081495]])

# L1 regularization ("sparse" - rzadsze wagi, tj. czestsze zerowanie sie wag)
lr = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
lr.fit(X_train_std, y_train)

print('Training accuaracy:', lr.score(X_train_std, y_train))
print('Test accuaracy:', lr.score(X_test_std, y_test))

# =============================================================================
# One-vs-Rest (OvR) approach by default where the first intercept belongs to the
# model that fits class 1 versus class 2 and 3; the second value is the intercept
# of the model that fits class 2 versus class 1 and 3; and the third value is
# the intercept of the model that fits class 3 versus class 1 and 2, respectively.
# =============================================================================

lr.intercept_ # array([-0.3838466 , -0.15808912, -0.70046231])
lr.coef_
# array([[ 0.28034402,  0.        ,  0.        , -0.0280363 ,  0.        ,
#          0.        ,  0.71007789,  0.        ,  0.        ,  0.        ,
#          0.        ,  0.        ,  1.23611397],
#        [-0.64382879, -0.06887818, -0.0571765 ,  0.        ,  0.        ,
#          0.        ,  0.        ,  0.        ,  0.        , -0.92682526,
#          0.06012522,  0.        , -0.37113029],
#        [ 0.        ,  0.0616476 ,  0.        ,  0.        ,  0.        ,
#          0.        , -0.63548339,  0.        ,  0.        ,  0.49793758,
#         -0.35818425, -0.5717131 ,  0.        ]])

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(penalty='l1',
                            C=10**c,
                            random_state=0,
                            solver='liblinear')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column+1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5.0), 10**5.0])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()

# %% Sequential feature selection algorithms
# Sequential Backward Selection (SBS)
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = estimator
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        # Podzielenie na zbior treningowy i testowy
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        # Przypisanie wymiaru    
        dim = X_train.shape[1]
        # utowrzenie "tuple'a", np. tuple(range(3)) => (0, 1, 2)
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []
            # "Przypisanie" do kazdego indeksu, wymiaru w jakim byla iteracja
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            # utworzenie tablic z wynikami i redukcja wymiaru o jeden
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self
    
    # Zwrocenie tych kolumn, gdzie odnotowano najlepszy wynik
    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train,
                    X_test, y_test, indices):
        # wyuczenie modelu na wskazanych indeksach
        self.estimator.fit(X_train[:, indices], y_train)
        # predykcja na wskazanych indeksach
        y_pred = self.estimator.predict(X_test[:, indices])
        # zapisanie wyniku na wskazanych indeksach
        score = self.scoring(y_test, y_pred)
        return score
    
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
# =============================================================================
# The SBS fit method will then create new training-subsets for testing
# (validation) and training, which is why this test set is also called
# validation dataset. This approach is necessary to prevent our original
# test set becoming part of the training data.
# =============================================================================

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuaracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# przejdzmy do osmej redukcji (patrz wykres od prawej strony)
k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])
# ['Alcohol', 'Malic acid', 'Alcalinity of ash', 'Hue', 'Proline']

# Original vs reduced
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))
# Training accuracy: 0.98 to Test accuracy: 0.94 indicate overfitting

knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:',
      knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:',
      knn.score(X_test_std[:, k5], y_test))
# Training accuracy: 0.95 to Test accuracy: 0.96 indicate lack of overfitting

# %% Assessing feature importance with random forest
# =============================================================================
# In the previous sections, we learned how to use L1 regularization to zero out
# irrelevant features via logistic regression and use the SBS algorithm for feature
# selection.
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,
                                  random_state=0,
                                  n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[f],
                            importances[indices[f]]))
    
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()