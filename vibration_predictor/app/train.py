import numpy as np
import Files
import extract_features
import random
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()

sampling_rate = 20e6  # signal sampling rate

folder = "E:\\Data\\Aleksandr\\IMS_data\\test1\\"
files = Files.list_ims_files(folder)
total_files = len(files)
train_files = int(0.8*total_files)  # 80-20 split

# Assign labels to data
labels = np.concatenate([np.zeros(train_files), np.ones(total_files - train_files)])
files_labels = list(zip(files, labels))

# Shuffle and separate train and test data
random.shuffle(files_labels)
files_shuffled, labels_shuffled = zip(*files_labels)
train_data = files_shuffled[:train_files]
train_labels = labels_shuffled[:train_files]

# Test data
test_data = files_shuffled[train_files:]
test_labels = labels_shuffled[train_files:]



X_train = []
Y_train = []


# Extract features from the files (training)
for ii in range(len(train_data)):
    data = Files.load_ims_file(train_data[ii])
    channel_N = data.shape[1]
    label = train_labels[ii]
    feat = []
    for jj in range(channel_N):
        tmp = extract_features.extract_features(data[:, jj], sampling_rate)
        feat.append(tmp)

    features_final = np.concatenate(feat, axis=0)
    X_train.append(features_final)
    Y_train.append(label)

X_train = np.vstack(X_train)
scaler.fit(X_train)

# Feed train data and fit the model
dtrain = xgb.DMatrix(X_train, label=Y_train)
model = xgb.train({"objective": "binary:logistic", "tree_method": "hist"}, dtrain)


# Test
Y_test = []
X_test = []

# Extract features for test
for ii in range(len(test_data)):
    data = Files.load_ims_file(test_data[ii])
    channel_N = data.shape[1]
    label = test_labels[ii]
    feat = []
    for jj in range(channel_N):
        tmp = extract_features.extract_features(data[:, jj], sampling_rate)
        feat.append(tmp)

    features_final = np.concatenate(feat, axis=0)
    X_test.append(features_final)
    Y_test.append(label)

# Feed test data and test
X_test = np.vstack(X_test)
X_test = scaler.transform(X_test)
dtest = xgb.DMatrix(X_test, label=Y_test)

# Prediction
pred = model.predict(dtest)
yhat = (pred > 0.5).astype(int)

# Save model
model.save_model("xgb_model.json")
joblib.dump(scaler, "app/model/scaler.joblib")
