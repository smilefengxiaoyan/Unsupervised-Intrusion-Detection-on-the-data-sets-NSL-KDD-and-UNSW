import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from joblib import load
from joblib import dump
from SVM import read_data


###  load true label
testing_df  = pd.read_csv("/Users/smile/Desktop/master paper/master project/KDD/NSL_KDDTest+.csv", header=None)

orig = np.array(testing_df)
orig_labels = orig[:, 41]
binary_labels = np.where(orig_labels == 'normal', 0, 1)

### load the model
ocsvm = load('ocsvm_model.joblib')
gmm = load('gmm_model.joblib')
# Load data
test_set, label_set = read_data("/Users/smile/Desktop/master paper/master project/KDD/processed_test_latent20_and_labels.csv")


# Data scaling
scaler = StandardScaler()
test_set = scaler.fit_transform(test_set)
latent_features = test_set
latent_label= label_set
ocsvm_preds = ocsvm.predict(latent_features)
# Calculate the score (log probability density) for each point under the GMM
gmm_scores = gmm.score_samples(latent_features)

# Determine a threshold to define what constitutes a "low" probability
probability_threshold = np.percentile(gmm_scores, 5)  # e.g., 5th percentile

# Boolean array where True indicates an outlier
verified_anomalies = (ocsvm_preds == -1) & (gmm_scores < probability_threshold)



# Convert `verified_anomalies` from a boolean to integer (0 and 1), where -1 (anomaly detected by OCSVM) becomes 1
predictions = np.where(verified_anomalies, 1, 0)

# Assuming `latent_label` contains 0s for normal and 1s for anomalies

true_labels = binary_labels

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

# False Positive Rate (FPR)
fpr = fp / (fp + tn)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("False Positive Rate:", fpr)



