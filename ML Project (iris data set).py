#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()


# In[3]:


dir(iris)


# In[4]:


iris.data[12]


# In[5]:


iris.feature_names


# In[6]:


df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head(5)


# In[7]:


iris.target_names


# In[8]:


df['target'] = iris.target
df.head(5)


# In[9]:


df[df.target== 1].head()


# In[10]:


df['flower_names'] = df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[11]:


df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]


# In[28]:



df1.head(3)


# In[30]:


df2.head(4)


# In[31]:


df0.head(3)


# In[13]:


import matplotlib.pyplot as plt 


# In[14]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


# In[15]:


plt.xlabel('petal Length')
plt.ylabel('petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')


# In[16]:


x = df.drop(['target','flower_names'], axis = 'columns')
x.head(5)


# In[17]:


y = df.target
y


# In[18]:


from sklearn.model_selection import train_test_split 


# In[19]:


X_train,X_test, y_train,y_test = train_test_split(x,y,test_size= 0.2)


# In[20]:


len(X_test)


# In[21]:


len(X_train)


# In[22]:


from sklearn.svm import SVC
model = SVC(C =100)


# In[23]:


model.fit(X_train,y_train)


# In[24]:


model.score(X_test,y_test)


# In[33]:


model.predict([[4.7,3.2,1.3,0.2]])


# In[ ]:




