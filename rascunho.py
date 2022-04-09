from pandas import read_csv
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from funcoes_de_processamento import k_fold
import pandas as pd

data = pd.read_csv("data.csv")
fold_sets = k_fold(data, 10)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
dt_classifier = DecisionTreeClassifier(random_state=0)

for fold_set in fold_sets:
    Xtraining = fold_set["training_data"]
    ytraining = fold_set["training_data"]
    #knn_classifier.fit(X,y)

#print(fold_set["training_data"])
