import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
from joblib import load
from joblib import dump
import time
from sklearn.svm import SVC

def read_data(filename):
    df = pd.read_csv(filename, skiprows=1, header=None)
    features = df.iloc[:, 1:-1].values  # Assuming the last column is the label
    labels = df.iloc[:, -1].values
    labels_transformed = -2 * labels + 1
    return features, labels_transformed

# Load data
train_set, label_set = read_data("/Users/smile/Desktop/master paper/master project/KDD/processed_latent_and_labels.csv")

# Data scaling
scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)

# Extract latent
latent_features = train_set
latent_label= label_set
start_time = time.time()
# Train OCSVM
#ocsvm = svm.SVC(kernel='linear', C=100)
ocsvm = OneClassSVM(kernel='rbf', nu=0.01,gamma=1.0)
#nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
# It is a way to control the sensitivity of the model to outliers and does not relate to the C parameter in SVC.
ocsvm.fit(latent_features)
svc_duration = time.time() - start_time



# Training GaussianMixture
start_time = time.time()
gmm = GaussianMixture(n_components=10, covariance_type='full')
gmm.fit(train_set)
gmm_duration = time.time() - start_time


print(f"Trained OCSVM in {svc_duration:.2f} seconds")
print(f"Trained GaussianMixture in {gmm_duration:.2f} seconds")





dump(ocsvm, 'ocsvm_model.joblib')
dump(gmm, 'gmm_model.joblib')



"""
#### test

ocsvm_preds = ocsvm.predict(latent_features)
# Calculate the score (log probability density) for each point under the GMM
gmm_scores = gmm.score_samples(latent_features)

# Determine a threshold to define what constitutes a "low" probability
probability_threshold = np.percentile(gmm_scores, 5)  # e.g., 5th percentile

# Boolean array where True indicates an outlier
verified_anomalies = (ocsvm_preds == -1) & (gmm_scores < probability_threshold)
# Plot data points, coloring by OCSVM predictions
plt.scatter(latent_features[:, 0], latent_features[:, 1], c=ocsvm_preds, cmap='coolwarm', alpha=0.5, marker='o')

# Highlight verified anomalies
anomaly_indices = np.where(verified_anomalies)
plt.scatter(latent_features[anomaly_indices, 0], latent_features[anomaly_indices, 1], color='red', label='Verified Anomalies')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Verification: OCSVM Predictions Refined by GMM')
plt.legend()
plt.show()

"""
