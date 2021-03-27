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
index=[]
target=[]
for i in range(456,570):
    index.append(i)
x_test= pd.read_csv('Tumor Cancer Prediction_test.csv', index_col=False,)
x_test.drop('Index', axis =1, inplace=True)
x_test = np.array(x_test)
###############################################################
scaler =StandardScaler()
x_test =scaler.fit_transform(x_test)

######################### Model #########################################
# load the model from disk 
filename = 'xgboost_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
start_time = time.time()
result = loaded_model.predict(x_test)
end_time = time.time()
execution_time = (end_time - start_time)
print("Testing time",execution_time)
####################################################################
for i in result:
    if i==0:
        target.append('B')
    else:
        target.append('M')
res=pd.DataFrame(index,columns=['Index'])
res['diagnosis']=target
res.to_csv('xgboost.csv',index=False)
