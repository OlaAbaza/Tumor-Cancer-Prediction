#lets plot the bar graph
%matplotlib inline
import matplotlib.pyplot as plt
#Training time
train=[ 0.0029997825622558594,0.0069959163665771484,0.06496191024780273]
plt.figure(figsize=(8,5))
plt.bar(['SVM','desionTree','Xgboost'],train,color=['salmon','r','g','b','orange'],label='TrainingTime')
plt.ylabel('TrainingTime')
plt.xlabel('Algortihms')


#Testing time 
test=[ 0.0010018348693847656,0.0009987354278564453,0.0010013580322265625]
plt.figure(figsize=(8,5))
plt.bar(['SVM','desionTree','Xgboost'],test,color=['salmon','r','g','b','orange'],label='TestingTime')
plt.ylabel('TestingTime')
plt.xlabel('Algortihms')
#####################################################Training time#######################################################
#Training time pca
train=[ 0.003996849060058594,0.0009975433349609375,0.1281285285949707]
plt.figure(figsize=(8,5))
plt.bar(['SVM','desionTree','Xgboost'],train,color=['salmon','r','g','b','orange'],label='TrainingTime')
plt.ylabel('TrainingTime _PCA')
plt.xlabel('Algortihms')


#Testing time pca
test=[ 0.0,0.0,0.0009996891021728516]
plt.figure(figsize=(8,5))
plt.bar(['SVM','desionTree','Xgboost'],test,color=['salmon','r','g','b','orange'],label='TestingTime')
plt.ylabel('TestingTime-PCA')
plt.xlabel('Algortihms')

############################################Accuarcy################################################################
#Accuarcy pca
train=[ 0.89473,0.85964,0.94736]
plt.figure(figsize=(8,5))
plt.bar(['SVM','desionTree','Xgboost'],train,color=['salmon','r','g','b','orange'],label='Accuarcy _PCA')
plt.ylabel('Accuarcy _PCA')
plt.xlabel('Algortihms')


#Accuarcy 
test=[ 0.94736,0.96491,0.96491]
plt.figure(figsize=(8,5))
plt.bar(['SVM','desionTree','Xgboost'],test,color=['salmon','r','g','b','orange'],label='Accuarcy')
plt.ylabel('Accuarcy')
plt.xlabel('Algortihms')
