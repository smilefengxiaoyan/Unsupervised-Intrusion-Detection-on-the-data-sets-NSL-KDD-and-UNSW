from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score,confusion_matrix


# This the baseline for GMM model

set = "UNSW"

# Load  NSL-KDD data
if set == "NSLKDD":
    train_set_orginal = pd.read_csv(
        "/Users/smile/Desktop/master paper/master project/KDD/processed_normal_train_feature.csv", header=None)
    test_set_orginal = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/processed_test_feature.csv",
                                   header=None)
    train_set_orginal = train_set_orginal.iloc[1:, 1:]
    test_set_orginal = test_set_orginal.iloc[1:, 1:]

    ###### Always the same
    test_labels = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/processed_test_label.csv",
                              header=None)
    labels = test_labels.iloc[1:, -1].values
    true_labels = [-2 * int(label) + 1 for label in labels]
    true_labels = np.array(true_labels)
    ######



######
else:
    train_set_orginal = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_normal_train_feature.csv",
                                    header=None)
    test_set_orginal = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_test_feature.csv",
                                   header=None)
    train_set_orginal = train_set_orginal.iloc[1:, 1:]
    test_set_orginal = test_set_orginal.iloc[1:, 1:]

    train_set_label = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_normal_train_label.csv")
    train_set_label = train_set_label.iloc[:, -1].values
    ###### Always the same
    test_labels = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_test_label.csv",
                              header=None)
    labels = test_labels.iloc[1:, -1].values
    true_labels = [-2 * int(label) + 1 for label in labels]
    true_labels = np.array(true_labels)







# Load  NSL-KDD data






# Data scaling  nslkdd already done
scaler = MinMaxScaler()

train_set = train_set_orginal
#option:test_set_AE30,test_set_orginal
test_set = test_set_orginal

# Training GaussianMixture
start_time = time.time()
gmm = GaussianMixture(n_components=10)
gmm.fit(train_set)
gmm_duration = time.time() - start_time

print(f"Trained GaussianMixture in {gmm_duration:.2f} seconds")












# Initialize StratifiedKFold
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays to store the accuracy, precision, recall, and F1-score for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
fpr_scores = []
# Perform cross-validation
for train_index, test_index in skf.split(test_set, true_labels):
    X_train, X_test = test_set.iloc[train_index], test_set.iloc[test_index]
    #X_train, X_test = test_set[train_index], test_set[test_index]
    y_train, y_test = true_labels[train_index], true_labels[test_index]

    y_test = 1 - (y_test + 1) / 2
    X_train_nomaly = X_train[y_train == 1]
    gmm.fit(X_train_nomaly)
    # Calculate GMM scores
    t_scores = -gmm.score_samples(X_train_nomaly)
    probability_threshold = np.percentile(t_scores, 90)

    gmm_scores = -gmm.score_samples(X_test)


    # Determine a threshold to define what constitutes a "low" probability
      # e.g., 5th percentile  for the negative

    # Boolean array where True indicates an outlier
    verified_anomalies = gmm_scores > probability_threshold



    # Convert verified_anomalies to the same format as true labels
    verified_anomaly_preds = (np.zeros(X_test.shape[0]))
    verified_anomaly_preds[verified_anomalies] = 1


    # Calculate the accuracy for the current fold
    correct_predictions = np.sum(verified_anomaly_preds == y_test)
    accuracy = correct_predictions / len(y_test)

    # Calculate precision, recall, and F1-score for the current fold
    precision = precision_score(y_test, verified_anomaly_preds)
    recall = recall_score(y_test, verified_anomaly_preds)
    f1 = f1_score(y_test, verified_anomaly_preds)

    # Store the scores for the current fold
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Calculate and print confusion matrix for the current fold
    conf_matrix = confusion_matrix(y_test, verified_anomaly_preds)
    print(f'Confusion Matrix for fold:')
    print(conf_matrix)
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fpr_scores.append(fpr)



# Calculate the mean accuracy, precision, recall, and F1-score across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)
mean_fpr = np.mean(fpr_scores)
print(f'Mean Cross-Validation Accuracy: {mean_accuracy:.4f}')
print(f'Mean Cross-Validation Precision: {mean_precision:.4f}')
print(f'Mean Cross-Validation Recall: {mean_recall:.4f}')
print(f'Mean Cross-Validation F1-Score: {mean_f1:.4f}')
print(f'Mean Cross-Validation FPR: {mean_fpr:.4f}')