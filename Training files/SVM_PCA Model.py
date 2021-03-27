#load libraries
import numpy as np        
from scipy.stats import randint
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import svm
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

####################################################################
x_train = np.array(X)
y_train = np.array(Y)

#X_val = np.array(X_val)
#y_val = np.array(y_val)
###################scaler########################################
scaler =StandardScaler()
x_train = scaler.fit_transform(x_train)
###########################################PCA###############################
#principle component analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.90)
x_train = pca.fit_transform(x_train)
explained_variance=pca.explained_variance_ratio_
pickle.dump(pca, open("svm.pca_Value","wb"))
####################################################################
clf1 = svm.SVC(gamma=0.01,kernel='linear')
start_time = time.time()
clf1.fit(x_train,y_train)
end_time = time.time()
execution_time = (end_time - start_time)
print("Training time",execution_time)

# save the model to disk
filename = 'svm_pca_model.sav'
pickle.dump(clf1, open(filename, 'wb'))
