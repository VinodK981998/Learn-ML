#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


df = pd.read_csv("D:/Roadmap/One Hot Encoding/homeprices.csv")


# In[10]:


dummy_vars = pd.get_dummies(df.town, drop_first = True) # dropped first variable cause of dummy variable trap

df = pd.concat([df,dummy_vars],axis='columns').drop('town',axis='columns')


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


regr = LinearRegression()
regr.fit(df.drop('price',axis=1),df['price'])


# In[25]:


regr.predict([[2800,1,0]])


# In[26]:


regr.predict([[3400,0,1]])


# In[19]:


df.drop('price',axis=1)


# In[27]:


regr.score(df.drop('price',axis=1),df['price'])  # R-square value, cofficient of determination


# In[33]:


#using inbulit methods to do the same
dfle = pd.read_csv("D:/Roadmap/One Hot Encoding/homeprices.csv")


# In[30]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[42]:


dfle.town = le.fit_transform(dfle.town)


# In[55]:


X = dfle.drop("price",axis=1).values
y= dfle.price


# In[36]:


from sklearn.preprocessing import OneHotEncoder


# In[61]:


from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')


# In[66]:


X = ct.fit_transform(X)
X = X[:,1:]


# In[67]:


regr.fit(X,y)


# In[69]:


regr.predict([[0,1,3400]]) # 3400 sqr ft home in west windsor


# In[ ]:




