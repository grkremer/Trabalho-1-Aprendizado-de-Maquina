from pandas import read_csv
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from funcoes_de_processamento import k_fold, split_validation_df
import pandas as pd

data = pd.read_csv("data.csv")


knn_classifier = KNeighborsClassifier(n_neighbors=3)
dt_classifier = DecisionTreeClassifier(random_state=0)
 
validation_df, model_df = split_validation_df(data)
#print(validation_df.describe())
#print(model_df.describe())
fold_sets = k_fold(model_df, 10)

features = data.columns
lista_features = features.to_list()
lista_features.remove("diagnosis")
lista_features.remove("id")
lista_features.remove("Unnamed: 32")

for fold_set in fold_sets:
    training_df = fold_set["training_data"]
    testing_df = fold_set["test_data"]

    Xtraining = training_df[lista_features]
    ytraining = training_df["diagnosis"]

print(f"Xtreino {training_df.describe()}")
print(f"ytreino {testing_df.describe()}")
print(f"dados {validation_df.describe()}")
#print(fold_set["training_data"])
