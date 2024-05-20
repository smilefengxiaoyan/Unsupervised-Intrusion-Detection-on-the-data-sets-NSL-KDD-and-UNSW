from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
def int_encode(dataset):
    features_list = dataset.select_dtypes(include='object')
    le = LabelEncoder()

    for x in features_list :
        dataset[x] = le.fit_transform(dataset[x])
    return dataset


def normalize_data(features):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features