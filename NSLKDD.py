import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

from time import time
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score,auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM as OCSVM

from sklearn.utils import shuffle

import pandas as pd
import os
import sys

def meanAUC_PRAUC(auc_list, pr_list, name):
    print('AUC:')
    print(auc_list)
    print('**************')
    print('PR AUC:')
    print(pr_list)
    print('**************')

    AUC_Frame = pd.DataFrame(auc_list, columns=[name])
    PR_AUC_Frame = pd.DataFrame(pr_list, columns=[name])

    AUC_Frame.to_csv(f'./auc/{name}.csv', index=False)
    PR_AUC_Frame.to_csv(f'./auc_PR/{name}.csv', index=False)

    AUC_Frame = list(AUC_Frame[name])

    N = len(AUC_Frame)
    mean_auc = np.mean(AUC_Frame)
    std_auc = np.std(AUC_Frame)
    std_error = std_auc / (np.sqrt(N))

    ci = 1.96 * std_error
    lower_bound = mean_auc - ci
    upper_bound = mean_auc + ci

    print('AUC')
    print(f'{mean_auc:.2f} +/- {ci:.2f}')
    print(f'95% confidence level, average auc would be between {lower_bound:.2f} and {upper_bound:.2f}')
    print('**************')

    PR_AUC_Frame = list(PR_AUC_Frame[name])

    N = len(PR_AUC_Frame)
    mean_auc = np.mean(PR_AUC_Frame)
    std_auc = np.std(PR_AUC_Frame)
    std_error = std_auc / (np.sqrt(N))

    ci = 1.96 * std_error
    lower_bound = mean_auc - ci
    upper_bound = mean_auc + ci

    print('PR AUC')
    print(f'{mean_auc:.2f} +/- {ci:.2f}')
    print(f'95% confidence level, average auc would be between {lower_bound:.2f} and {upper_bound:.2f}')


def plot_pr_curve(precision, recall):
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def pr_auc(y_test, y_pred):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auc_score = auc(recall, precision)
    print(f'PR AUC: {auc_score:.2f}')
    plot_pr_curve(precision, recall)
    return auc_score
def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs"""
    y = df[target]
    x = df.drop(columns=target)
    return x, y


def _encode_text_dummy(df, name):
    names = []
    dummies = pd.get_dummies(df.loc[:, name])
    i = 0

    tmpL = []
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
        names.append(dummy_name)
        _x = [i, x]
        tmpL.append(_x)
        i += 1

    df.drop(name, axis=1, inplace=True)
    return names, tmpL


def get_NSLKDD( seed, mx=0.889, mz=0.028, my=0.083, scale=True, show=False):
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_hot_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'unknown']
    PATH_TRAIN = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv'
    PATH_TEST = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'
    train = pd.read_csv(PATH_TRAIN, delimiter=',', header=None, names=columns)
    test = pd.read_csv(PATH_TEST, delimiter=',', header=None, names=columns)

    train.drop(columns=['unknown'], inplace=True)
    test.drop(columns=['unknown'], inplace=True)

    rest = set(train.columns) - set(test.columns)
    for i in rest:
        idx = train.columns.get_loc(i)
        test.insert(loc=idx, column=i, value=0)

    df = pd.concat((train, test))
    discreteCol = ['protocol_type', 'service', 'flag']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['label'].copy()
    labels[labels != 'normal'] = 0  # anomalous
    labels[labels == 'normal'] = 1  # normal

    df['label'] = labels
    normal = df[df['label'] == 1]
    abnormal = df[df['label'] == 0]

    normal = shuffle(normal, random_state=seed)
    abnormal = shuffle(abnormal, random_state=seed)

    abnormal_1 = abnormal[:int(len(abnormal) * .5) + 1]
    abnormal_2 = abnormal[int(len(abnormal) * .5) + 1:]

    train_set = normal[:int(mx * len(normal))]
    val_normal = normal[int(mx * len(normal)): int(mx * len(normal)) + int(mz * len(normal))]
    test_normal = normal[int(mx * len(normal)) + int(mz * len(normal)):]

    val_abnormal = abnormal_1[:int(mz * len(normal))]
    test_abnormal = abnormal_1[int(mz * len(normal)):int(mz * len(normal)) + int(my * len(normal)) + 1]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(
            f'Training set is composed by {len(x_train[y_train == 1])} normal records and {len(x_train[y_train == 0])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(
            f'Validation set is composed by {len(x_val[y_val == 1])} normal records and {len(x_val[y_val == 0])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(
            f'Test set is composed by {len(x_test[y_test == 1])} normal records and {len(x_test[y_test == 0])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []

        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(columns) - len(discreteCol) - 1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset


def process_and_save_NSLKDD(seed, scale=True):
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_hot_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'unknown']

    PATH_TRAIN = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv'
    PATH_TEST = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'

    # Load the data
    train = pd.read_csv(PATH_TRAIN, delimiter=',', header=None, names=columns)
    test = pd.read_csv(PATH_TEST, delimiter=',', header=None, names=columns)

    # Drop the 'unknown' column
    train.drop(columns=['unknown'], inplace=True)
    test.drop(columns=['unknown'], inplace=True)

    # Ensure train and test have the same columns
    rest = set(train.columns) - set(test.columns)
    for i in rest:
        idx = train.columns.get_loc(i)
        test.insert(loc=idx, column=i, value=0)

    df = pd.concat((train, test))
    discreteCol = ['protocol_type', 'service', 'flag']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['label'].copy()
    labels[labels != 'normal'] = 1  # abnomal
    labels[labels == 'normal'] = 0  # normal
    labels = labels.astype(float)
    df['label'] = labels

    # Split the processed DataFrame back into train and test sets
    train_processed = df.iloc[:len(train), :]
    test_processed = df.iloc[len(train):, :]

    x_train, y_train = _to_xy(train_processed, target='label')
    x_test, y_test = _to_xy(test_processed, target='label')

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    # Save processed train and test sets to CSV
    train_processed_df = pd.DataFrame(x_train)[:-1]
    train_processed_df['label'] = y_train
    test_processed_df = pd.DataFrame(x_test)[:-1]
    test_processed_df['label'] = y_test

    train_processed_df.to_csv('./processed_train_original.csv', index=False)
    test_processed_df.to_csv('./processed_test_original.csv', index=False)
    print("Processed original train and test datasets have been saved as CSV files.")




# Example usage
process_and_save_NSLKDD(seed=42, scale=True)
