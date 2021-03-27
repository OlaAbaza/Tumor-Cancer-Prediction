#load libraries
import numpy as np        
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost
from xgboost import XGBClassifier
import time
import pickle

data_train = pd.read_csv('Tumor Cancer Prediction_train.csv', index_col=False,)
data_train.drop('Index', axis =1, inplace=True)
data_train['diagnosis'] = data_train['diagnosis'].map({'M':1,'B':0})

####################################split###################################
X=np.array(data_train.drop(['diagnosis'],1))
Y=np.array(data_train['diagnosis'])
#x_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=1) 
#print("Training data",x_train.shape)
#print("validation data",X_val.shape)

########################################################################

x_train = np.array(X)
y_train = np.array(Y)

#X_val = np.array(X_val)
#y_val = np.array(y_val)

###################scaler########################################
scaler =StandardScaler()
x_train = scaler.fit_transform(x_train)

####################################################################

classifier=xgboost.XGBClassifier(max_depth=3,learning_rate=0.10)

#clf = XGBClassifier(n_estimators=100)
start_time = time.time()
classifier.fit(x_train,y_train)
end_time = time.time()
execution_time = (end_time - start_time)
print("Training time",execution_time)

filename = 'xgboost_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

