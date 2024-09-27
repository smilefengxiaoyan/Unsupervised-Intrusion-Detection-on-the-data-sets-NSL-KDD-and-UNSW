import time
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score,confusion_matrix

# here ist for the experiment im superviesde method
set = "UNSW"

# Load  NSL-KDD data
if set == "NSLKDD":
    train_set_orginal  = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/processed_train_feature.csv", header=None)
    test_set_orginal  = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/processed_test_feature.csv", header=None)
    train_set_orginal = train_set_orginal.iloc[1:, 1:]
    test_set_orginal = test_set_orginal.iloc[1:, 1:]

    train_set_label= pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/processed_train_label.csv")
    train_set_label =train_set_label.iloc[:, -1].values
    ###### Always the same
    test_labels =  pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/processed_test_label.csv", header=None)
    labels =test_labels.iloc[1:, -1].values
    true_labels = [-2 * int(label) + 1 for label in labels]
    true_labels  = np.array(true_labels)
######
else:
    train_set_orginal = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_train_feature.csv",
                                    header=None)
    test_set_orginal = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_test_feature.csv",
                                   header=None)
    train_set_orginal = train_set_orginal.iloc[1:, 1:]
    test_set_orginal = test_set_orginal.iloc[1:, 1:]

    train_set_label = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_train_label.csv")
    train_set_label = train_set_label.iloc[:, -1].values
    ###### Always the same
    test_labels = pd.read_csv("/Users/smile/Desktop/master paper/master project/UNSW/UNSWprocessed_test_label.csv",
                              header=None)
    labels = test_labels.iloc[1:, -1].values
    true_labels = [-2 * int(label) + 1 for label in labels]
    true_labels = np.array(true_labels)




# Data scaling  nslkdd already done
scaler = MinMaxScaler()

train_set =train_set_orginal
#option:test_set_AE30,test_set_orginal
test_set = test_set_orginal



clf_SVM_Df=SVC(kernel='linear', C=1.0, random_state=0)
train0 = time.time()
clf_SVM_Df.fit(train_set, train_set_label.astype(int))
train1 = time.time() - train0

print(train1)

# Create Decision Tree classifer object
clf_Naive = GaussianNB()
train0 = time.time()
# Train Decision Tree Classifer
clf_Naive = clf_Naive.fit(train_set, train_set_label.astype(int))
train1 = time.time() - train0

print(train1)

clf_Tree = DecisionTreeClassifier()
train0 = time.time()
#Train Decision Tree Classifer
clf_Tree = clf_Tree.fit(train_set, train_set_label.astype(int))
train1 = time.time() - train0
print(train1)

# Initialize StratifiedKFold
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays to store the accuracy, precision, recall, and F1-score for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
fpr_scores = []
fpr_scores = []
# Perform cross-validation
for train_index, test_index in skf.split(test_set, true_labels):
    X_train, X_test = test_set.iloc[train_index], test_set.iloc[test_index]
    #X_train, X_test = test_set[train_index], test_set[test_index]
    y_train, y_test = true_labels[train_index], true_labels[test_index]



    X_train_nomaly = X_train[y_train == 1]

    X_train_abnomaly = X_train[y_train == -1]
    #This is the step i mentioned in the thesie to select the random 100 attacks
    ab =X_train_abnomaly.iloc[:100,]

    train_x = pd.concat([X_train_nomaly, ab])
    normal_labels = pd.Series(np.zeros(X_train_nomaly.shape[0]))
    abnormal_labels = pd.Series(np.ones(ab.shape[0]))

    # Concatenate the labels
    train_y = pd.concat([normal_labels, abnormal_labels])



    y_test = 1 - (y_test + 1) / 2



    clf_SVM_Df.fit(train_x, train_y.astype(int))
    #clf_Naive.fit(X_train_nomaly, Y_train_nomaly.astype(int))
    #clf_Tree.fit(X_train_nomaly, Y_train_nomaly.astype(int))
    #verified_anomaly_preds = clf_Naive.predict(X_test)
    verified_anomaly_preds=clf_SVM_Df.predict(X_test)
    #verified_anomaly_preds = clf_Tree.predict(X_test)


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



