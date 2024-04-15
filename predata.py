from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def integer_encode(dataset):
    features_list = dataset.select_dtypes(include='object')
    OHE = OneHotEncoder()
    for i in features_list:
        dataset[i] = OHE.fit_transform(dataset[i])
    return dataset


def normalize_dataset(features):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features