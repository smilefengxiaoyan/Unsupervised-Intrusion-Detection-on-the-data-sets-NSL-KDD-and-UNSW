import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Load NSL-KDD dataset
def load_nsl_kdd_dataset_train():
    # Feature names of the dataset
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
    ]

    # Read data from CSV file
    df = pd.read_csv('/Users/smile/Desktop/master paper/master project/KDD/KDDTrain+.txt', header=None, names=feature_names)

    return df

def load_nsl_kdd_dataset_test():
    # Feature names of the dataset
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
    ]

    # Read data from CSV file
    df = pd.read_csv('/Users/smile/Desktop/master paper/master project/KDD/KDDTest+.txt', header=None, names=feature_names)

    return df

def attack_type_file():
    df = pd.read_csv('/Users/smile/Desktop/master paper/master project/KDD/Attack Types.csv', header=None,names=['name', 'attack_type'])

    return df

def print_label_dist(label_col):
    c = Counter(label_col)
    print(f'label is {c}')


def get_feature_values_and_frequency(feature):
    # 统计每个值的出现次数
    value_counts = feature.value_counts()

    # 按照出现频率排序
    sorted_values = value_counts.sort_values(ascending=False)

    return sorted_values




# Main program
if __name__ == "__main__":
    # Load data
    nsl_kdd_train = load_nsl_kdd_dataset_train()
    nsl_kdd_test =  load_nsl_kdd_dataset_test()
    attack_type_df = attack_type_file()
    attack_type_dict = dict(zip(attack_type_df['name'].tolist(), attack_type_df['attack_type'].tolist()))

    nsl_kdd_train['attack_type'].replace(attack_type_dict, inplace=True) # change attack type to 5 class
    nsl_kdd_test['attack_type'].replace(attack_type_dict, inplace=True)

    print_label_dist(nsl_kdd_train['attack_type'])
    print_label_dist(nsl_kdd_test['attack_type'])

    nsl_kdd_train['attack_type'] = ['unknown' if label not in ['normal', 'dos', 'probe', 'r2l', 'u2r'] else label for
                                    label in nsl_kdd_train['attack_type']]
    nsl_kdd_test['attack_type'] = ['unknown' if label not in ['normal', 'dos', 'probe', 'r2l', 'u2r'] else label for
                                   label in nsl_kdd_test['attack_type']]

    train_label = nsl_kdd_train['attack_type']
    train_label['type'] = 'train'
    test_label = nsl_kdd_test['attack_type']
    test_label['type'] = 'test'
    train_label = nsl_kdd_train[['attack_type']].copy()
    train_label['type'] = 'train'
    test_label = nsl_kdd_test[['attack_type']].copy()
    test_label['type'] = 'test'

    # Combine the labels and reset the index
    label_all = pd.concat([train_label, test_label], axis=0).reset_index(drop=True)

    # Calculate counts for each attack type in train and test sets
    label_counts = label_all.groupby(['attack_type', 'type']).size().unstack().fillna(0)



    # Sum totals for each dataset type
    total_train_test = label_counts.sum(axis=0)

    # Calculate the percentage of each attack type within each dataset
    label_percent = label_counts.div(total_train_test, axis=1) * 100

    # Define the desired order for the categories
    categories = ['normal', 'dos', 'probe', 'r2l', 'u2r', 'unknown']

    # Convert 'attack_type' into a categorical type with a defined order
    label_percent['attack_type'] = pd.Categorical(label_percent.index, categories=categories, ordered=True)

    # Sort the DataFrame by the new categorical type
    label_percent.sort_index(inplace=True)
    print(label_percent.index)

    # Plotting code (assuming label_percent has already been set up for plotting)
    fig, ax = plt.subplots(figsize=(14, 8))
    ind = range(len(label_percent))
    width = 0.35

    # Creating bar plots
    train_bars = ax.bar(ind, label_percent['train'], width, label='Train', color='b')
    test_bars = ax.bar([p + width for p in ind], label_percent['test'], width, label='Test', color='r')

    # Labels, title, and ticks
    ax.set_xlabel('Attack Type', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.set_title('Distribution of Attack Types in Training and Test Sets in NSLKDD', fontsize=16)
    ax.set_xticks([p + width / 2 for p in ind])
    ax.set_xticklabels(label_percent.index, rotation=45, ha='right')

    # Legend and grid
    ax.legend(title='Dataset Type', fontsize=12)
    ax.grid(axis='y', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.show()

    # Filter the training dataset for 'normal' attacks only
    #normal_train_data = nsl_kdd_train[nsl_kdd_train['attack_type'] == 'normal']

    # Save the filtered data to a new CSV file
    #normal_train_data.to_csv('normal_train_data_nsl.csv', index=False)