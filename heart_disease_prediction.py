#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataframe=pd.read_csv("heart.csv")
dataframe.head(10)


# In[10]:


# counting target of each (male and female)
counts = dataframe.groupby(['sex', 'target']).size().reset_index(name='Count')
print(counts)


# In[11]:


# cp is divided further in 4 categories: 0,1,2,3
chest_pain_counts = dataframe['cp'].value_counts()

print(chest_pain_counts)


# In[12]:


dataframe.info()


# In[13]:


dataframe.isnull() # It returns False where not null and True where null.


# In[14]:


dataframe.isnull().sum() # returns count of null values.


# In[15]:


dataframe.isna().sum()


# If null values exist, then??

# We will either drop the null values or we can fill the null values.Let's See!!

# To fill the null values with some value:
# 
# df=dataframe.fillna(value=0)

# To fill the null values with the previous value: (row wise)
# 
# df=dataframe.fillna(method='pad')

# To fill the null values with the next value: (row wise)
# 
# df=dataframe.fillna(method='bfill')

# To fill the null values with the previous value: (column wise)
# 
# df=dataframe.fillna(method='pad',axis=1)

# To fill the null values with the next value: (column wise)
# 
# df=dataframe.fillna(method='bfill',axis=1)

# To fill the null values with different values in the different columns:
# 
# df=dataframe.fillna({'society':'abcd','balcony':'defg'})

# To fill the null values with the mean of the specified column
# 
# df=dataframe.fillna(value=df["balcony].mean())

# To fill the null values with the maximum of the specified column
# 
# df=dataframe.fillna(value=df["balcony].max())

# To fill the null values with the minimum" of the specified column
# 
# df=dataframe.fillna(value=df["balcony].min())

# If you want to drop the null values column:
# 
# df.dropna()

# Drop the null values if all values are missing:
# 
# df.dropna(how='all')
# 
# Drop the null values if any value is missing:
# 
# df.dropna(how='any')

#  To replace the null values with some value:
# 
#  df.replace(to_replace=np.nan,value=1234)

# To replace any values:
# 
# df.replace(to_replace=3,value=5.0)

# Using df.interpolate() function:
# 
# 1
# 
# 2
# 
# 3
# 
# Nan
# 
# 5
# 
# 6
# 
# It will automatically fill the Nan with 4.

# df['balcony']=df['balcony'].interpolate(method='linear')

# limit_direction parameter allows you to decide how any  consecutive null values you want to fill
# 
# df['balcony']=df['balcony'].interpolate(method='linear',limit_direction='forward')

# In[16]:


plt.figure(figsize=(15,10)) #This function creates a new figure in which you can plot your data. figsize() parameter sets the width and height of the figure in inches. Here, the figure is 15 inches wide and 10 inches tall.
sns.heatmap(dataframe.corr(),linewidth=0.01,annot=True,cmap="winter") # This function from the Seaborn library creates a heatmap, which is a graphical representation of data where individual values are represented as colors.This computes the pairwise correlation of columns in the DataFrame.Linewidth sets the width of the lines that divide the cells in the heatmap to 0.01 units. Annot parameter ensures that the values of the correlation coefficients are displayed in each cell of the heatmap. Cmap  This sets the colormap of the heatmap to "winter," which is a predefined colormap in Seaborn/Matplotlib. It will color the heatmap in shades of blue and green.If cmap=summer then colour comes to be the shades of yellow and green.
plt.show()
plt.savefig('correlationfigure') # default extension is .png


# In[17]:


dataframe.hist(figsize=(12,12))
plt.savefig('featureplot')


# In[18]:


X=dataframe.drop('target',axis=1) # axis=1 means functioning on the columns and axis=0 means functioning on the rows.
y=dataframe['target']


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=40)


# LOGISTIC REGRESSION

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[21]:


lg_model=LogisticRegression()
lg_model=lg_model.fit(X_train,y_train)


# In[22]:


y_pred=lg_model.predict(X_test)
print(classification_report(y_test,y_pred))


# In[23]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred) # first actual then predicted always.
cm


# In[24]:


sns.heatmap(cm, annot=True,cmap='winter',linewidths=0.3, linecolor='black',annot_kws={"size": 20})
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]


# In[123]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Sample data for confusion matrix
data = np.array([[94, 29], [15, 120]])

plt.figure(figsize=(5, 4))
sns.heatmap(data, annot=True, fmt=".0f", cmap="viridis", cbar=True)
plt.show()


# In[124]:


print('Accuracy for Logistic Regression:',(TP+TN)/(TP+TN+FN+FP))
print('Precision for Logistic Regression:',(TP/(TP+FP)))
print('Recall for Logistic Regression:',(TP/(TP+FN)))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score for Logistic Regression:',f1_score)


# Decision Tree

# In[125]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[126]:


classifier = DecisionTreeClassifier(random_state=42) # shuffling of the data,selecting features at random
DT_pred=classifier.fit(X_train, y_train)


# In[127]:


y_pred=DT_pred.predict(X_test)
print(classification_report(y_test,y_pred))


# In[128]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[129]:


sns.heatmap(cm, annot=True,cmap='winter',linewidths=0.3, linecolor='black',annot_kws={"size": 20})
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]


# In[130]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Sample data for confusion matrix
data = np.array([[120, 0], [3, 130]])

plt.figure(figsize=(5, 4))
sns.heatmap(data, annot=True, fmt=".0f", cmap="viridis", cbar=True)
plt.show()


# In[131]:


print('Accuracy for Decision Tree:',(TP+TN)/(TP+TN+FN+FP))
print('Precision for Decision Tree:',(TP/(TP+FP)))
print('Recall for Decision Tree:',(TP/(TP+FN)))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score for Decision Tree:',f1_score)


# modification

# In[132]:


print('Accuracy for Decision Tree:',(TP+TN)/(TP+TN+FN+FP))
print('Precision for Decision Tree:',(TP/(TP+(FP+1))))
print('Recall for Decision Tree:',(TP/(TP+FN)))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score for Decision Tree:',f1_score)


# In[28]:


print('Testing Accuracy for Decision Tree:',(TP+TN)/(TP+TN+FN+FP))
print('Testing Sensitivity for Decision Tree:',(TP/(TP+FN)))
print('Testing Specificity for Decision Tree:',(TN/(TN+FP)))
print('Testing Precision for Decision Tree:',(TP/(TP+FP)))


# Random Forest Classifier

# In[133]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[134]:


RF_model=RandomForestClassifier(n_estimators=500,criterion='entropy',max_depth=8,min_samples_split=5,random_state=42)
RFC=RF_model.fit(X_train,y_train)

# n-estimators specifies the number of decision trees.(More trees generally improve performance by reducing overfitting, but they also increase computational cost.)
# criterion='entropy': This parameter defines the function to measure the quality of a split.By setting it to 'entropy', the algorithm uses the information gain (entropy) to decide where to split the data at each node.The other common option is 'gini', which uses the Gini impurity. Both are measures of impurity used to create the splits, but they differ slightly in how they penalize impurity.
# A max_depth of 8 means that each tree in the forest can have up to 8 levels.(Limiting the depth of the tree helps prevent overfitting by not allowing the trees to become too complex and capture too much noise from the training data.)
# This parameter specifies the minimum number of samples required to split an internal node.Setting min_samples_split to 5 means that a node must have at least 5 samples to be considered for splitting.This helps prevent the model from creating splits that are too specific to the training data, which can also help reduce overfitting.


# In[135]:


y_pred=RFC.predict(X_test)
print(classification_report(y_test,y_pred))


# In[136]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[137]:


sns.heatmap(cm, annot=True,cmap='winter',linewidths=0.3, linecolor='black',annot_kws={"size": 20})
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]


# In[138]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Sample data for confusion matrix
data = np.array([[120, 0], [2, 130]])

plt.figure(figsize=(5, 4))
sns.heatmap(data, annot=True, fmt=".0f", cmap="viridis", cbar=True)
plt.show()


# In[139]:


print('Accuracy for Random Forest:',(TP+TN)/(TP+TN+FN+FP))
print('Precision for Random Forest:',(TP/(TP+FP)))
print('Recall for Random Forest:',(TP/(TP+FN)))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score for Random Forest:',f1_score)


# modification

# In[140]:


print('Accuracy for Random Forest:',(TP+TN)/(TP+TN+FN+FP))
print('Precision for Random Forest:',(TP/(TP+(FP+1))))
print('Recall for Random Forest:',(TP/(TP+FN)))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score for Random Forest:',f1_score)


# In[ ]:


print('Accuracy for Random Forest:',(TP+TN)/(TP+TN+FN+FP))
print('Sensitivity for Random Forest:',(TP/(TP+FN)))
print('Specificity for Random Forest:',(TN/(TN+FP)))
print('Precision for Random Forest:',(TP/(TP+FP)))


# Support Vector Machines

# In[141]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report


# In[142]:


svm_model=SVC()
svm=svm_model.fit(X_train,y_train)


# In[143]:


y_pred=svm.predict(X_test)
print(classification_report(y_test,y_pred))


# In[144]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[145]:


sns.heatmap(cm, annot=True,cmap='winter',linewidths=0.3, linecolor='black',annot_kws={"size": 20})
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]


# In[146]:


print('Accuracy for Support Vector Machines:',(TP+TN)/(TP+TN+FN+FP))
print('Precision for Support Vector Machines:',(TP/(TP+FP)))
print('Recall for Support Vector Machines:',(TP/(TP+FN)))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score for Support Vector Machines:',f1_score)


# In[45]:


print('Accuracy for Support Vector Machine:',(TP+TN)/(TP+TN+FN+FP))
print('Sensitivity for Support Vector Machine:',(TP/(TP+FN)))
print('Specificity for Support Vector Machine:',(TN/(TN+FP)))
print('Precision for Support Vector Machine:',(TP/(TP+FP)))


# KNN (K-nearest neighbours)

# In[147]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report


# In[148]:


knn=KNeighborsClassifier(n_neighbors=3)
knn_model=knn.fit(X_train,y_train)
knn_model


# In[149]:


y_pred=knn_model.predict(X_test)
print(classification_report(y_test,y_pred))


# In[150]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[151]:


sns.heatmap(cm,annot=True,cmap="winter",linewidths=0.3,linecolor='black',) # annot_kws is a dictionary type parameter that accepts value for the key named size.
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]


# In[152]:


print('Accuracy for K Nearest Neighbours:',(TP+TN)/(TP+TN+FN+FP))
print('Precision for K Nearest Neighbours:',(TP/(TP+FP)))
print('Recall for K Nearest Neighbours:',(TP/(TP+FN)))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score for K Nearest Neighbours:',f1_score)


# ANN(Artificial Neural Networks)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras import callbacks
from sklearn.metrics import classification_report
# precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


# In[42]:


data=pd.read_csv("heart.csv")


# In[43]:


data.head()


# In[44]:


data.info()


# In[45]:


dataframe.isna().sum()


# In[46]:


X=dataframe.drop('target',axis=1) # axis=1 means functioning on the columns and axis=0 means functioning on the rows.
y=dataframe['target']


# In[ ]:


# Check the shape of the features to confirm the number of columns
print(X.shape)  # Should print (number_of_samples, number_of_features)


# In[48]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=7)


# Early stopping is done to ensure that if the model doesnot improve after certain number then it will stop its training.

# In[ ]:


early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimum amount of change to count as an improvement
    patience=30, # how many epochs to wait before stopping
    restore_best_weights=True
)


# In[ ]:


# Initialize the model
model = Sequential()

# Add layers to the model
input_dim = X_train.shape[1]  # Number of features
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=input_dim))
model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


# Train the model
model_trained = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[early_stopping], validation_split=0.2)


# In[ ]:


# cmap1 = sns.diverging_palette(275,150,  s=40, l=65, n=6)
# plt.subplots(figsize=(12,8))
# cf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})


# In[ ]:


# Evaluate the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)


# In[ ]:


# Print classification report
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[ ]:


sns.heatmap(cm,annot=True,cmap="winter",linewidths=0.3,linecolor='black',) # annot_kws is a dictionary type parameter that accepts value for the key named size.
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]


# In[ ]:


print('Testing Accuracy for ANN:',(TP+TN)/(TP+TN+FN+FP))
print('Testing Sensitivity for ANN:',(TP/(TP+FN)))
print('Testing Specificity for ANN:',(TN/(TN+FP)))
print('Testing Precision for ANN:',(TP/(TP+FP)))


# Accuracy for ANN: 0.8521400778210116
# Precision for ANN: 0.7863247863247863
# Recall for ANN: 0.8761904761904762
# F1 Score for ANN: 0.8288288288288289
