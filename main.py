import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import LabelEncoder

import predata as prep
import AE as AE

dataset = "NSLKDD_laten"


if dataset == "UNSWNB15":

    training_df = pd.read_csv('/Users/smile/Desktop/master paper/master project/UNSW/Training and Testing Sets/UNSW_NB15_training-set.csv')
    testing_df = pd.read_csv('/Users/smile/Desktop/master paper/master project/UNSW/Training and Testing Sets/UNSW_NB15_testing-set.csv')

    training_df = training_df.dropna()
    testing_df = testing_df.dropna()

    orig = np.array(testing_df)
    orig_labels = orig[0:50000, 43]

    training_df = prep.int_encode(training_df)
    testing_df = prep.int_encode(testing_df)

    training_data = np.array(training_df)
    testing_data = np.array(testing_df)

    training_labels = training_data[0:150000, 43]
    training_features = training_data[0:150000, 0:43]

    testing_labels = testing_data[0:50000, 43]
    testing_features = testing_data[0:50000, 0:43]

    # Perform normalization on dataset
    training_features = prep.normalize_data(training_features)
    testing_features = prep.normalize_data(testing_features)

    inputs_size= 43
    hidden_size = [38, 30, 38]
    output_size = 43
    num_train =20
    train_size = 10000

    lr = 0.001
    active_f = F.elu

    print("\nPerforming Feature Selection... \n")
    # Selecting features from chosen dataset


    M_type ="Simple"
    print("\nSimple model... \n")
    select_fea_si = []
    select_fea_si = AE.training(M_type, training_features, lr, active_f, inputs_size, hidden_size,  output_size, training_df,num_train,train_size)
    select_fea_si.append(43)
    rank = len(select_fea_si)


    M_type = "AEDropout"
    print("\nWith Dropout model... \n")
    select_fea = []
    select_fea = AE.training(M_type, training_features, lr, active_f, inputs_size, hidden_size, output_size,
                             training_df, num_train, train_size)
    select_fea.append(43)
    rank = len(select_fea)


    result_df = pd.DataFrame({
        'simple': select_fea_si,
        'dropout': select_fea
    })

    # Save the DataFrame to CSV, without the index
    result_df.to_csv('UNSWselected_features.csv', index=False)



elif dataset == "NSLKDD":

    training_df = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/NSL_KDDTrain+.csv", header=None)
    testing_df = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/NSL_KDDTest+.csv", header=None)

    training_df = training_df.dropna()
    testing_df = testing_df.dropna()

    orig = np.array(testing_df)
    orig_labels = orig[0:40000, 41]

    training_df = prep.int_encode(training_df)
    testing_df = prep.int_encode(testing_df)

    training_data = np.array(training_df)
    testing_data = np.array(testing_df)

    training_labels = training_data[0:120000, 41]
    training_features = training_data[0:120000, 0:41]

    testing_labels = testing_data[0:40000, 41]
    testing_features = testing_data[0:40000, 0:41]

    # Perform normalization on dataset
    training_features = prep.normalize_data(training_features)
    testing_features = prep.normalize_data(testing_features)

    inputs_size = 41
    hidden_size = [35, 30, 35]
    output_size = 41
    num_train = 20
    train_size = 10000



    lr = 0.001
    active_f = nn.ELU

    print("\nPerforming Feature Selection... \n")

    M_type = "Simple"
    print("\nSimple model... \n")

    # Selecting features from chosen dataset
    select_fea_si = []

    select_fea_si = AE.training(M_type, training_features, lr, active_f, inputs_size, hidden_size, output_size,training_df,num_train,train_size)

    select_fea_si.append(41)
    rank= len(select_fea_si)

    M_type = "AEDropout"
    print("\nWith Dropout model... \n")

    # Selecting features from chosen dataset
    select_fea = []

    select_fea = AE.training(M_type, training_features, lr, active_f, inputs_size, hidden_size, output_size,
                             training_df, num_train, train_size)

    select_fea.append(41)
    rank = len(select_fea)

    result_df = pd.DataFrame({
        'simple': select_fea_si,
        'dropout': select_fea
        })

    # Save the DataFrame to CSV, without the index
    #result_df.to_csv('Nslkddselected_features.csv', index=False)



elif dataset == "NSLKDD_laten":

    training_df = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/NSL_KDDTrain+.csv", header=None)
    testing_df = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/NSL_KDDTest+.csv", header=None)

    training_df = training_df.dropna()
    testing_df = testing_df.dropna()

    orig = np.array(testing_df)
    orig_labels = orig[0:40000, 41]

    training_df = prep.int_encode(training_df)
    testing_df = prep.int_encode(testing_df)

    training_data = np.array(training_df)
    testing_data = np.array(testing_df)

    training_labels = training_data[0:120000, 41]
    training_features = training_data[0:120000, 0:41]

    testing_labels = testing_data[0:40000, 41]
    testing_features = testing_data[0:40000, 0:41]

    # Perform normalization on dataset
    training_features = prep.normalize_data(training_features)
    testing_features = prep.normalize_data(testing_features)

    inputs_size = 41
    hidden_size = [35, 30, 35]
    output_size = 41
    num_train = 20
    train_size = 10000
    M_type= None



    lr = 0.001
    active_f = nn.ELU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE.AE(M_type, training_features, lr, active_f, inputs_size, hidden_size, output_size, training_df,num_train,train_size)
    model.load_state_dict(torch.load('./trained_model.pth'))
    model.to(device)
    model.eval()





    train_errors = AE.calculate_reconstruction_errors(train_dataset, model)
    threshold = np.percentile(train_errors, 95)  # 95th percentile as threshold

# Detecting anomalies on new data
    test_errors = AE.calculate_reconstruction_errors(test_dataset, model)
    anomalies = test_errors > threshold

# Visualization of the threshold and errors
    plt.hist(test_errors, bins=50, alpha=0.75, label='Test Errors')
    plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    plt.title('Reconstruction errors with Anomaly Threshold')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Print anomaly detection results
    print("Detected Anomalies:", np.sum(anomalies))