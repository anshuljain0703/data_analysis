#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies

# In[26]:


#Here we are predicting whether the object is a rock or a mine by using logistic regression


# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Data Collection and Data Processing

# In[2]:


sonar_data=pd.read_csv('sonar data.csv',header=None)


# In[3]:


sonar_data.head()


# In[4]:


sonar_data.shape


# In[5]:


sonar_data.describe()


# In[6]:


sonar_data[60].value_counts()


# In[9]:


sonar_data.groupby(60).mean()


# In[11]:


# separating data and Labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]


# In[13]:


print(X)
print(Y)


# # Spliting the data

# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.1, stratify=Y, random_state=1)


# In[15]:


print(X_train)
print(Y_train)


# # Model Training --> Logistic Regression

# In[18]:


model=LogisticRegression()


# In[19]:


model.fit(X_train,Y_train)


# # Model Evaluation

# In[20]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 


# In[21]:


print(training_data_accuracy)


# In[22]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 


# In[23]:


print(test_data_accuracy)


# # Making a Predictive System

# In[27]:


input_data = (0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a mine')


# In[ ]:




