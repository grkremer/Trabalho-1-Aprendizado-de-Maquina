from pandas import read_csv
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from funcoes_de_processamento import k_fold, split_validation_df
import pandas as pd

data = pd.read_csv("healthcare-dataset-stroke-data.csv")



def fix_undersampling(data, feature, under_value=1):
    under_count = data[feature].where(data[feature] == under_value).count()
    greater_count = data[feature].where(data[feature] != under_value).count()
    while greater_count > under_count:
        print(greater_count)
        item = data.sample()
        while item[feature].where(data[feature] == under_value).count() == 0:
            item = data.sample()
        data.drop(item.index, inplace=True)
        greater_count -= 1
    return data

def fix_undersampling2(data, feature, under_value=1):
    under_count = data[feature].where(data[feature] == under_value).count()
    print("under count: " + str(under_count))
    apenas_under = data.where(data[feature] == under_value).dropna(how='all')
    apenas_not_under = data.where(data[feature] != under_value).dropna(how='all')
    print(data.count())
    print(apenas_under.count())
    print(apenas_not_under.count())
    item = apenas_not_under.sample()
    apenas_not_under.drop(item.index, inplace=True)
    dados = item
    for i in range(1, under_count):
        item = apenas_not_under.sample()
        apenas_not_under.drop(item.index, inplace=True)
        dados = pd.concat([dados,item])
    dados = pd.concat([dados, apenas_under])
    return dados

dados = fix_undersampling2(data,"stroke",1)
print(dados.count())