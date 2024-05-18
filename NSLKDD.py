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
    df = pd.read_csv('/Users/smile/Desktop/maste paper/master project/KDD/KDDTrain+.txt', header=None, names=feature_names)

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
    df = pd.read_csv('/Users/smile/Desktop/maste paper/master project/KDD/KDDTest+.txt', header=None, names=feature_names)

    return df

def attack_type_file():
    df = pd.read_csv('/Users/smile/Desktop/maste paper/master project/KDD/Attack Types.csv', header=None,names=['name', 'attack_type'])

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
    label_all = pd.concat([train_label, test_label], axis=0)
    print(label_all)
    print(test_label)
    sns.countplot(x='attack_type', hue='type', data=label_all)




    # Print basic information of the dataset
    print("Basic information of NSL-KDD dataset:")
    print(nsl_kdd_train.info())
    print(nsl_kdd_test.info())





    # Print descriptive statistics of the dataset
    print("\nDescriptive statistics of NSL-KDD dataset:")
    print(nsl_kdd_df.describe())

    # Count distribution of attack types
    print("\nDistribution of attack types:")
    print(nsl_kdd_df['attack_type'].value_counts())

    unknown_attacks = nsl_kdd_df[nsl_kdd_df['attack_type'] == 'unknown']
    print("Number of samples with unknown attack types:", len(unknown_attacks))
    print("Characteristics of samples with unknown attack types:")
    print(unknown_attacks.head())
    # Count distribution of different difficulty levels
    print("\nDistribution of difficulty levels:")
    print(nsl_kdd_df['difficulty_level'].value_counts())

    # Count frequency of attack types
    attack_type_counts = nsl_kdd_df['attack_type'].value_counts()

    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(attack_type_counts, labels=attack_type_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Attack Types Distribution')
    plt.show()

    ###### Date situation
    missing_values = nsl_kdd_df.isnull().sum()
    print("missing values：\n", missing_values)

    threshold = 3  # 3 mal std
    outliers = nsl_kdd_df[(nsl_kdd_df - nsl_kdd_df.mean()).abs() > threshold * nsl_kdd_df.std()].dropna()
    print("outliers：\n", outliers)


