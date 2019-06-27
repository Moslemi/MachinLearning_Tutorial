#!/usr/bin/env python
# coding: utf-8

# # Supervised Learning and K Nearest Neighbors Exercises

# ## Introduction
# 
# We will be using customer churn data from the telecom industry for this week's exercises. The data file is called 
# `Orange_Telecom_Churn_Data.csv`. We will load this data together, do some preprocessing, and use K-nearest neighbors to predict customer churn based on account characteristics.

# In[1]:


from __future__ import print_function
import os
data_path = ['data']


# ## Question 1
# 
# * Begin by importing the data. Examine the columns and data.
# * Notice that the data contains a state, area code, and phone number. Do you think these are good features to use when building a machine learning model? Why or why not? 
# 
# We will not be using them, so they can be dropped from the data.

# In[2]:


import pandas as pd

# Import the data using the file path
filepath = os.sep.join(data_path + ['Orange_Telecom_Churn_Data.csv'])
data = pd.read_csv(filepath)


# In[17]:


data.head(1).T


# In[4]:


# Remove extraneous columns    axis = 1 colums and axis = 0 index
data.drop(['state', 'area_code', 'phone_number'], axis=1, inplace=True)


# In[5]:


data.columns


# ## Question 2
# 
# * Notice that some of the columns are categorical data and some are floats. These features will need to be numerically encoded using one of the methods from the lecture.
# * Finally, remember from the lecture that K-nearest neighbors requires scaled data. Scale the data using one of the scaling methods discussed in the lecture.

# In[6]:


from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

for col in ['intl_plan', 'voice_mail_plan', 'churned']:
    data[col] = lb.fit_transform(data[col])


# In[7]:


# Mute the sklearn warning
import warnings
warnings.filterwarnings('ignore', module='sklearn')

from sklearn.preprocessing import MinMaxScaler

msc = MinMaxScaler()

data = pd.DataFrame(msc.fit_transform(data),  # this is an np.array, not a dataframe.
                    columns=data.columns)


# ## Question 3
# 
# * Separate the feature columns (everything except `churned`) from the label (`churned`). This will create two tables.
# * Fit a K-nearest neighbors model with a value of `k=3` to this data and predict the outcome on the same data.

# In[9]:


# Get a list of all the columns that don't contain the label
x_cols = [x for x in data.columns if x != 'churned']

# Split the data into two dataframes
X_data = data[x_cols]
y_data = data['churned']

# # alternatively:
# X_data = data.copy()
# y_data = X_data.pop('churned')


# In[20]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn = knn.fit(X_data, y_data)

y_pred = knn.predict(X_data)


# ## Question 4
# 
# Ways to measure error haven't been discussed in class yet, but accuracy is an easy one to understand--it is simply the percent of labels that were correctly predicted (either true or false). 
# 
# * Write a function to calculate accuracy using the actual and predicted labels.
# * Using the function, calculate the accuracy of this K-nearest neighbors model on the data.

# In[15]:


# Function to calculate the % of values that were correctly predicted

def accuracy(real, predict):
    return sum(y_data == y_pred) / float(real.shape[0])


# In[16]:


print(accuracy(y_data, y_pred))


# ## Question 5
# 
# * Fit the K-nearest neighbors model again with `n_neighbors=3` but this time use distance for the weights. Calculate the accuracy using the function you created above. 
# * Fit another K-nearest neighbors model. This time use uniform weights but set the power parameter for the Minkowski distance metric to be 1 (`p=1`) i.e. Manhattan Distance.
# 
# When weighted distances are used for part 1 of this question, a value of 1.0 should be returned for the accuracy. Why do you think this is? *Hint:* we are predicting on the data and with KNN the model *is* the data. We will learn how to avoid this pitfall in the next lecture.

# In[ ]:


#Student writes code here


# In[21]:


knn = KNeighborsClassifier(n_neighbors=3 , weights = 'distance')

knn = knn.fit(X_data, y_data)

y_pred = knn.predict(X_data)
print(accuracy(y_data, y_pred))


# In[23]:


knn = KNeighborsClassifier(n_neighbors=3 , weights = 'uniform', p=1)

knn = knn.fit(X_data, y_data)

y_pred = knn.predict(X_data)
print(accuracy(y_data, y_pred))


# ## Question 6
# 
# * Fit a K-nearest neighbors model using values of `k` (`n_neighbors`) ranging from 1 to 20. Use uniform weights (the default). The coefficient for the Minkowski distance (`p`) can be set to either 1 or 2--just be consistent. Store the accuracy and the value of `k` used from each of these fits in a list or dictionary.
# * Plot (or view the table of) the `accuracy` vs `k`. What do you notice happens when `k=1`? Why do you think this is? *Hint:* it's for the same reason discussed above.

# In[ ]:


#Student writes code here


# In[53]:


def accuracy(real, predict):
    return sum(y_data == y_pred) / float(real.shape[0])
with open('storage','w') as st:
    st.write("accuracy"+ " k" + "\n")
    for k in range(1,20,1):
        knn = KNeighborsClassifier(n_neighbors=k , weights = 'uniform', p=1)

        knn = knn.fit(X_data, y_data)

        y_pred = knn.predict(X_data)
        #print(accuracy(y_data,y_pred))
        
        st.write(str(accuracy(y_data,y_pred))+" "+ str(k)+"\n")


# In[68]:


df = pd.read_csv('storage',header = 0, delimiter = " ")
df.head()


# In[79]:


df.columns
#df['accuracy']
import matplotlib.pyplot as plt
df.groupby('k').max()
ax = plt.axes()

ax.scatter(df.accuracy, df.k)

# Label the axes
ax.set(xlabel='k',
       ylabel='accuracy',
       title='accuracy via kneighbor');


# In[70]:





# In[ ]:





# In[ ]:




