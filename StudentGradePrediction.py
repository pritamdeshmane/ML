#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # load CSV file

# In[164]:


df=pd.read_csv("Desktop//student-mat1.csv")


# In[165]:


df.head(10)


# In[166]:


df.shape


# In[167]:


#checking missing values if any
df.isnull().sum()


# In[168]:


#checking datatypes
df.dtypes


# In[169]:


df['school'].value_counts()


# In[170]:


df['address'].value_counts()


# In[171]:


df['famsize'].value_counts()


# In[172]:


df['Pstatus'].value_counts()


# In[175]:


df['guardian'].value_counts()


# In[176]:


dict={'M':1 , 'F':0,
    'U':1 , 'R':0,
     'GT3':1 , 'LE3':0,
     'GP':1 , 'MS':0,
     'T':1 , 'A':1,
      'other':1,'services':2,'at_home':3,'teacher':4,'health':5,
      'no':0, 'yes':1,
      'course':1,'home':2,'reputation':3,'other':4,
      'mother':1,'father':2,'other':3,
      
     }

encode_dict={'sex':dict,
            'address':dict,
            'famsize': dict,
            'school':dict,
            'Pstatus':dict,
            'Mjob':dict,
            'Fjob':dict,
            'schoolsup':dict,
            'famsup':dict,
            'paid':dict,
            'activities':dict,
            'nursery':dict,
            'higher':dict,
            'internet':dict,'romantic':dict,
            'reason':dict,'guardian':dict}

df.replace(encode_dict, inplace=True)


# In[177]:


df.head(10)


# In[178]:


df.dtypes


# In[179]:


df.head()


# In[184]:


x=df.drop(['G3'],axis=1)
y=df['G3']
x.shape,y.shape


# # Train_test_split
# 

# In[189]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=123)


# In[190]:


from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.metrics import mean_squared_error as mse


# In[223]:


reg=knn(n_neighbors=10)

reg.fit(train_x,train_y)
test_predict=reg.predict(test_x)
k=mse(test_predict,test_y)
print('test MSE ', k)


# In[ ]:


reg.predict()

