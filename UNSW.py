import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing

# This is for Data prepossing for the Dataset UNSW


PATH_TRAIN1 = '/Users/smile/Desktop/master paper/master project/UNSW/Training and Testing Sets/UNSW_NB15_training-set.csv'
PATH_TRAIN2 = '/Users/smile/Desktop/master paper/master project/UNSW/Training and Testing Sets/UNSW_NB15_testing-set.csv'

df = pd.read_csv(PATH_TRAIN1 )
df_test = pd.read_csv(PATH_TRAIN2)

for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


categorical_columns=['proto', 'service', 'state']

df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]

df_categorical_values.head()

# protocol type
unique_protocol=sorted(df.proto.unique())
string1 = 'proto'
unique_protocol2=[string1 + x for x in unique_protocol]
print(unique_protocol2)

# service
unique_service=sorted(df.service.unique())
string2 = 'service'
unique_service2=[string2 + x for x in unique_service]
print(unique_service2)


# flag
unique_flag=sorted(df.state.unique())
string3 = 'state'
unique_flag2=[string3 + x for x in unique_flag]
print(unique_flag2)


# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2


#do it for test set
unique_state_test=sorted(df_test.state.unique())
unique_state2_test=[string3 + x for x in unique_state_test]

unique_proto_test=sorted(df_test.proto.unique())
unique_proto2_test=[string1 + x for x in unique_proto_test]


testdumcols=unique_proto2_test + unique_service2 + unique_state2_test

df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)



# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

enc = OneHotEncoder(categories='auto')
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)


# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)


trainstate=df['state'].tolist()
teststate= df_test['state'].tolist()
differencestate=list(set(trainstate) - set(teststate))
stringstate = 'state'
differencestate=[stringstate + x for x in differencestate]

for col in differencestate:
    testdf_cat_data[col] = 0

differencestatetrain=list( set(teststate)-set(trainstate))
differencestatetrain=[stringstate + x for x in differencestatetrain]
for col in differencestatetrain:
    df_cat_data[col] = 0




trainservice=df['proto'].tolist()
testservice= df_test['proto'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'proto'
difference=[string + x for x in difference]

for col in difference:
    testdf_cat_data[col] = 0






newdf=df.join(df_cat_data)
newdf.drop('state', axis=1, inplace=True)
newdf.drop('proto', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
newdf.drop('attack_cat', axis=1, inplace=True)
# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('state', axis=1, inplace=True)
newdf_test.drop('proto', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
newdf_test.drop('attack_cat', axis=1, inplace=True)

newdf_columns = newdf.columns
newdf_test = newdf_test[newdf_columns]


# put the new label column back
dfnoml = newdf[newdf['label'] == 0]

X_Df = dfnoml.drop('label',1)
Y_Df = dfnoml.label


#X_Df = newdf.drop('label',1)
#Y_Df = newdf.label

# test set
X_Df_test = newdf_test.drop('label',1)
Y_Df_test = newdf_test.label




scaler1 = preprocessing.MinMaxScaler().fit(X_Df)
X_Df=scaler1.transform(X_Df)

# test data
scaler5 = preprocessing.MinMaxScaler().fit(X_Df_test)
X_Df_test=scaler5.transform(X_Df_test)

df_X = pd.DataFrame(X_Df)
df_Y= pd.DataFrame(Y_Df)
df_X_test = pd.DataFrame(X_Df_test)
df_Y_test= pd.DataFrame(Y_Df_test)



df_X.to_csv("UnSWprocessed_normal_train_feature.csv")
df_Y.to_csv("UNSWprocessed_normal_train_label.csv")
df_X.to_csv("UnSWprocessed_train_feature.csv")
df_Y.to_csv("UNSWprocessed_train_label.csv")
df_X_test.to_csv("UNSWprocessed_test_feature.csv")
df_Y_test.to_csv("UNSWprocessed_test_label.csv")







