import matplotlib.pyplot as plt
# here the file for generate the figure for the masterthesie
# Data for plotting
np_thresholds = ['np70', 'np80', 'np90']
accuracy = [0.8952, 0.8369, 0.8289]
f1_score = [0.9052, 0.8345, 0.8499]
precision = [None, 0.9877, None]
recall = [None, 0.7225, None]
fpr = [0.0839, 0.0119, 0.2004]

# Creating the plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Accuracy
axs[0, 0].plot(np_thresholds, accuracy, marker='o', color='b')
axs[0, 0].set_title('Accuracy by NP Threshold')
axs[0, 0].set_xlabel('NP Threshold')
axs[0, 0].set_ylabel('Accuracy')

# Plot 2: F1-Score
axs[0, 1].plot(np_thresholds, f1_score, marker='o', color='g')
axs[0, 1].set_title('F1-Score by NP Threshold')
axs[0, 1].set_xlabel('NP Threshold')
axs[0, 1].set_ylabel('F1-Score')

# Plot 3: Precision
axs[1, 0].plot(np_thresholds, precision, marker='o', color='r')
axs[1, 0].set_title('Precision by NP Threshold')
axs[1, 0].set_xlabel('NP Threshold')
axs[1, 0].set_ylabel('Precision')

# Plot 4: Recall
axs[1, 1].plot(np_thresholds, recall, marker='o', color='m')
axs[1, 1].set_title('Recall by NP Threshold')
axs[1, 1].set_xlabel('NP Threshold')
axs[1, 1].set_ylabel('Recall')

# Adjusting layout
plt.tight_layout()
plt.show()



# Data for the GMM model, OCSVM+GMM, My Model, and Old Model with different thresholds

# Threshold levels
thresholds = ['np80', 'np85', 'np90', 'np95', 'np99']

# Performance metrics for each model
gmm_accuracy = [0.8900, 0.9054, 0.9117, 0.9147, 0.8541]
gmm_f1_score = [0.9086, 0.9195, 0.9225, 0.9218, 0.8529]
gmm_fpr = [0.2028, 0.1524, 0.1027, 0.0547, 0.0161]

ocsvm_gmm_accuracy = [0.8410, 0.8995, 0.8952, 0.8823, 0.8656]
ocsvm_gmm_f1_score = [0.8715, 0.9110, 0.9052, 0.8886, 0.8675]
ocsvm_gmm_fpr = [0.2996, 0.1061, 0.0839, 0.0460, 0.0142]

my_model_accuracy = [0.8302, 0.8625, 0.8681, 0.8424, 0.6702]
my_model_f1_score = [0.8659, 0.8860, 0.8855, 0.8510, 0.6091]
my_model_fpr = [0.3397, 0.2377, 0.1714, 0.0915, 0.0434]

old_model_accuracy = [0.8275, 0.8479, 0.8365, 0.8362, 0.7460]
old_model_f1_score = [0.8511, 0.8661, 0.8522, 0.8435, 0.7239]
old_model_fpr = [0.2235, 0.1732, 0.1525, 0.0828, 0.0423]

# Plotting Accuracy, F1-Score, and FPR for different threshold levels
plt.figure(figsize=(18, 5))

# Accuracy Plot
plt.subplot(1, 3, 1)
plt.plot(thresholds, gmm_accuracy, marker='o', linestyle='-', label='GMM Model', color='#FF8C00')
plt.plot(thresholds, ocsvm_gmm_accuracy, marker='o', linestyle='-', label='OCSVM + GMM', color='#1E90FF')
plt.plot(thresholds, my_model_accuracy, marker='o', linestyle='-', label='My Model', color='#32CD32')
plt.plot(thresholds, old_model_accuracy, marker='o', linestyle='--', label='Old Model', color='#FF4500')
plt.xlabel('Threshold Levels')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()
plt.grid(True)

# F1-Score Plot
plt.subplot(1, 3, 2)
plt.plot(thresholds, gmm_f1_score, marker='o', linestyle='-', label='GMM Model', color='#FF8C00')
plt.plot(thresholds, ocsvm_gmm_f1_score, marker='o', linestyle='-', label='OCSVM + GMM', color='#1E90FF')
plt.plot(thresholds, my_model_f1_score, marker='o', linestyle='-', label='My Model', color='#32CD32')
plt.plot(thresholds, old_model_f1_score, marker='o', linestyle='--', label='Old Model', color='#FF4500')
plt.xlabel('Threshold Levels')
plt.ylabel('F1-Score')
plt.title('F1-Score Comparison')
plt.legend()
plt.grid(True)

# FPR Plot
plt.subplot(1, 3, 3)
plt.plot(thresholds, gmm_fpr, marker='o', linestyle='-', label='GMM Model', color='#FF8C00')
plt.plot(thresholds, ocsvm_gmm_fpr, marker='o', linestyle='-', label='OCSVM + GMM', color='#1E90FF')
plt.plot(thresholds, my_model_fpr, marker='o', linestyle='-', label='My Model', color='#32CD32')
plt.plot(thresholds, old_model_fpr, marker='o', linestyle='--', label='Old Model', color='#FF4500')
plt.xlabel('Threshold Levels')
plt.ylabel('FPR')
plt.title('FPR Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
