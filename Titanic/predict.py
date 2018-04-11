import numpy as np
import pandas as pd
from keras.models import load_model

from data_perspactive import MinMaxScaler, encode_categories

def from_softmax_to_cates(matrix):
    cates = list()
    for irow in range(matrix.shape[0]):
        vector = matrix[irow, :].reshape(matrix.shape[1], )
        cates.append(vector.argmax())
    return cates

model = load_model('models\\8268.h5')
test = pd.read_csv('dataset\\test.csv')

test_norm = pd.DataFrame()
test_norm['PassengerId'] = test['PassengerId']

pclass = np.array(test['Pclass'])
mms_pclass = MinMaxScaler()
mms_pclass.fit(pclass)
pclass = mms_pclass.transform(pclass)
test_norm['Pclass'] = pclass

sex = np.array(test['Sex'])
sex = encode_categories(sex)
test_norm['Sex'] = sex

sibsp = np.array(test['SibSp'])
mms_sibsp = MinMaxScaler()
mms_sibsp.fit(sibsp)
sibsp = mms_sibsp.transform(sibsp)
test_norm['SibSp'] = sibsp

parch = np.array(test['Parch'])
mms_parch = MinMaxScaler()
mms_parch.fit(parch)
parch = mms_parch.transform(parch)
test_norm['Parch'] = parch

fare = np.array(test['Fare'])
mms_fare = MinMaxScaler()
mms_fare.fit(fare)
fare = mms_fare.transform(fare)
test_norm['Fare'] = fare

embarked = np.array(test['Embarked'])
embarked = encode_categories(embarked)
# Scale encoded embarked into range 0-1
mms_embarked = MinMaxScaler()
mms_embarked.fit(embarked)
embarked = mms_embarked.transform(embarked)
test_norm['Embarked'] = embarked

X_test = np.array(test_norm.iloc[:, 1:])
y_predict = model.predict(X_test)
y_predict = from_softmax_to_cates(y_predict)

output = pd.DataFrame()
output['PassengerId'] = test['PassengerId']
output['Survived'] = y_predict
output.to_csv('output.csv', index=None)