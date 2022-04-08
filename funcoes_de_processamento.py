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
            data.drop(item.index)
            folds[fold_index].append(item)
            if data.empty:
                break
    return folds

def k_fold(data, k=5):
    data_clone = deepcopy(data)
    shuffle(data_clone)
    folds = split_in_folds(data_clone, k)
    fold_sets = []
    for fold_index, fold in enumerate(folds):
        test_fold = fold
        training_folds = deepcopy(folds)
        training_folds = training_folds[:fold_index]+training_folds[fold_index:]
        training_folds_df = []
        for training_fold in training_folds:
            training_fold_df = training_fold[0]
            for training_item_index in range(1, len(training_fold)):
                pd.concat(training_fold_df, training_fold[training_item_index])
            training_folds_df.append(training_fold_df)
        fold_sets.append({'training_folds': training_folds_df, 'test_fold': test_fold})
    return fold_sets

def get_confusion_matrix(tp, fn, fp, tn):
    return {'tp':tp, 'fn':fn, 'fp':fp, 'tn':tn}

def get_sensitivity(confusion_matrix):
    return confusion_matrix['tp']/(confusion_matrix['tp']+confusion_matrix['fn'])

