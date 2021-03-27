import numpy as np 
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.stats import randint
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import svm
import time
from scipy.stats import norm
import seaborn as sns # visualization
#############################################################################################
data_train = pd.read_csv('Tumor Cancer Prediction_train.csv', index_col=False,)
data_train.drop('Index', axis =1, inplace=True)

sns.countplot(x='diagnosis',data=data_train)
plt.show()

print(data_train.isnull().any())
data_train['diagnosis'] = data_train['diagnosis'].map({'M':1,'B':0})
##############################analysis###############################
#basic descriptive statistics
data_train.describe()
corr = data_train.corr()  
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['diagnosis']>0.5)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data_train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

for i in data_train:
    if i != 'diagnosis':
        sns.catplot(x='diagnosis',y=i,data=data_train,kind='box')
