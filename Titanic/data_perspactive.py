import os

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Scale sequence value into a limited range.
class MinMaxScaler(object):
    def __init__(self):
        self.maximum = 0
        self.minimum = 0

    def fit(self, seq):
        self.maximum = max(seq)
        self.minimum = min(seq)

    def transform(self, seq, output_range=(0, 1)):
        seq_transformed = list()
        for value in seq:
            seq_transformed.append(
                output_range[0] + (output_range[1] - output_range[0]) *
                (value - self.minimum) / (self.maximum - self.minimum))
        return np.array(seq_transformed)


# Encode different categories into scalar
def encode_categories(seq):
    seen = list()
    seq_transformed = list()
    for value in seq:
        if value not in seen:
            seen.append(value)
        else:
            pass
        seq_transformed.append(seen.index(value))
    return np.array(seq_transformed)


# Load dataset
train = pd.read_csv('dataset\\train.csv')
# test = pd.read_csv('dataset\\test.csv')

# Data normalization
df_norm = pd.DataFrame()
df_norm['PassengerId'] = train[
    'PassengerId']  # PassengerId, no need to normalize
df_norm['Survived'] = train[
    'Survived']  # Survived or not, no need to normalize

# Scale pclass into range 0-1
pclass = np.array(train['Pclass'])
mms_pclass = MinMaxScaler()
mms_pclass.fit(pclass)
pclass = mms_pclass.transform(pclass)
df_norm['Pclass'] = pclass

# Encode sex into 0, 1
sex = np.array(train['Sex'])
sex = encode_categories(sex)
df_norm['Sex'] = sex

# Age: some empty value, skip for now

# Scale SibSp into range 0-1
sibsp = np.array(train['SibSp'])
mms_sibsp = MinMaxScaler()
mms_sibsp.fit(sibsp)
sibsp = mms_sibsp.transform(sibsp)
df_norm['SibSp'] = sibsp

# Scale Parch into range 0-1
parch = np.array(train['Parch'])
mms_parch = MinMaxScaler()
mms_parch.fit(parch)
parch = mms_parch.transform(parch)
df_norm['Parch'] = parch

# Ticket no.: not clear how to use it, skip for now

# Scale fare into range 0-1
fare = np.array(train['Fare'])
mms_fare = MinMaxScaler()
mms_fare.fit(fare)
fare = mms_fare.transform(fare)
df_norm['Fare'] = fare

# Cabin: too many empty values, skip for now

# Encode embarked into 0, 1, 2
embarked = np.array(train['Embarked'])
embarked = encode_categories(embarked)
# Scale encoded embarked into range 0-1
mms_embarked = MinMaxScaler()
mms_embarked.fit(embarked)
embarked = mms_embarked.transform(embarked)
df_norm['Embarked'] = embarked


# Build Neural Network for predicting
## Create train and test set randomly
X = np.array(df_norm.iloc[:, 2:])
y = np.array(df_norm['Survived']).reshape(-1, 1)
y = OneHotEncoder(n_values=2).fit_transform(y)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

optimizer = SGD(lr=0.02, decay=1e-5)
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=6))
model.add(Dense(5, activation='sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X, y, batch_size=64, epochs=3000, validation_split=0.2)


'''
82.09% --> sigmoid*2, sgd, batchsize=32, validation=0.3
82.68% --> sigmoid*2, SGD(lr=0.02, decay=1e-5), batchsize=64, validation=0.2
'''
