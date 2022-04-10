from copy import deepcopy
from itertools import count
import random
from matplotlib.pyplot import axis
from sklearn.utils import shuffle
from math import ceil
import pandas as pd

def printa(a):
    print(a)
    return

def normalizaMinMax(coluna):
    minimo = coluna.min()
    maximo = coluna.max()
    return (coluna-minimo)/(maximo-minimo)

def split_in_folds(data, k):
    max_fold_size = ceil(data["id"].count()/k)
    folds = []
    for fold_index in range(k):
        folds.append([])
        for item_index in range(max_fold_size):
            item = data.sample()
            data.drop(item.index, inplace=True)
            folds[fold_index].append(item)
            if data.empty:
                break
    return folds

def merge_folds(folds):
    result = []
    for fold in folds:
        result += fold
    return result

def fold_to_df(fold):
    df = fold[0]
    for item_index in range(1, len(fold)):
        df = pd.concat([df, fold[item_index]])
    return df

def k_fold(data, k=5):
    data_clone = deepcopy(data)
    folds = split_in_folds(data_clone, k)
    fold_sets = []
    for fold_index, fold in enumerate(folds):
        test_fold = fold_to_df(fold)
        training_folds = deepcopy(folds)
        training_folds = training_folds[:fold_index]+training_folds[fold_index+1:]
        training_folds = merge_folds(training_folds)
        training_folds = fold_to_df(training_folds)
        fold_sets.append({'training_data': training_folds, 'test_data': test_fold})
    return fold_sets

def split_validation_df(data, percentage=0.1):
    model_df = deepcopy(data)
    validation_size = int(model_df["id"].count()*percentage)
    item = model_df.sample()
    validation_df = item
    model_df.drop(item.index, inplace=True)
    for item_index in range(1, validation_size):
        item = model_df.sample()
        validation_df = pd.concat([validation_df, item])
        model_df.drop(item.index, inplace=True)
    return validation_df, model_df

def fix_undersampling(data, feature, under_value=1):
    n_samples = data[feature].where(data[feature] == under_value).count()
    while data[feature].where(data[feature] != under_value).count() > n_samples:
        item = data.sample()
        while item[feature] == under_value:
            item = data.sample()
        data.drop(item.index, inplace=True)
    return data


#def split_in_folds2(data, k):
#    max_fold_size = ceil(data["id"].count()/k)
#    folds = []
#    for fold_index in range(k):
#        folds.append([])
#        for item_index in range(max_fold_size):
#            item = data.sample()
#            id = item["id"]
#            data.drop(item.index)
#            folds[fold_index].append(id)
#            if data.empty:
#                break
#    return folds

#def k_fold2(data, k=5):
#    data_clone = deepcopy(data)
#    shuffle(data_clone)
#    folds = split_in_folds(data_clone, k)
#    fold_sets = []
#    for fold_index, fold in enumerate(folds):
#        test_fold = fold
#        training_folds = deepcopy(folds)
#        training_folds = training_folds[:fold_index]+training_folds[fold_index+1:]
#       fold_sets.append({'training_folds': training_folds, 'test_fold': test_fold})
#    return fold_sets

def get_confusion_matrix(results:list, expected:list):
    tp = fn = fp = tn = 0
    for index, result in enumerate(results):
        if result == 1 and expected[index] == 1:
            tp += 1
        elif result == 0 and expected[index] == 0:
            tn += 1
        elif result == 1 and expected[index] == 0:
            fp += 1
        elif result == 0 and expected[index] == 1:
            fn += 1
    return {'tp':tp, 'fn':fn, 'fp':fp, 'tn':tn}

def get_recall(confusion_matrix):
    return confusion_matrix['tp']/(confusion_matrix['tp']+confusion_matrix['fn'])

def get_precision(confusion_matrix):
    return confusion_matrix['tp']/(confusion_matrix['tp']+confusion_matrix['fp'])

def get_accuracy(confusion_matrix):
    return (confusion_matrix['tp']+confusion_matrix['tn'])/(confusion_matrix['tp']+confusion_matrix['tn']+confusion_matrix['fp']+confusion_matrix['fn'])

def get_f1_score(confusion_matrix):
    return (2*confusion_matrix['tp'])/(2*confusion_matrix['tp']+confusion_matrix['fp']+confusion_matrix['fn'])