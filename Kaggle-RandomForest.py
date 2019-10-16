#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Read both dataset and pre-process together
data_train = pd.read_csv("E:/Trinity/Machine Learning/Kaggle/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv")
data_test = pd.read_csv("E:/Trinity/Machine Learning/Kaggle/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv")


# In[3]:


data = pd.concat([data_train, data_test], sort=False)


# In[4]:


data.head()


# In[5]:


len(data)


# In[6]:


data = data.drop("Instance", axis=1)     # Drop column that has no relevance


# In[7]:


data.isnull().sum()                      # Check dataframe for NULL values


# In[8]:


# Rename all multi-word column names to single words for easier access

data = data.rename(index=str, columns={"Body Height [cm]" : "Height"})
data = data.rename(index=str, columns={"Year of Record" : "YearOfRecord"})
data = data.rename(index=str, columns={"Size of City" : "SizeOfCity"})
data = data.rename(index=str, columns={"University Degree": "UniversityDegree"})
data = data.rename(index=str, columns={"Wears Glasses" : "WearsGlasses"})
data = data.rename(index=str, columns={"Hair Color" : "HairColor"})
data = data.rename(index=str, columns={"Income in EUR" : "Income"})


# In[9]:


data.YearOfRecord.unique()


# In[10]:


data.Gender.unique()


# In[11]:


# Data imputation

data['Gender'] = data['Gender'].replace('0', "other")                   
data['Gender'] = data['Gender'].replace('unknown', pd.np.nan) 


# In[12]:


data.Age.unique()


# In[13]:


data.UniversityDegree.unique()


# In[14]:


# Label Encoding for University Degree giving different weights to different degrees
data['UniversityDegree'] = data['UniversityDegree'].replace('PhD', 4) 
data['UniversityDegree'] = data['UniversityDegree'].replace('Master', 3) 
data['UniversityDegree'] = data['UniversityDegree'].replace('Bachelor', 2) 
data['UniversityDegree'] = data['UniversityDegree'].replace('No', 0) 
data['UniversityDegree'] = data['UniversityDegree'].replace(pd.np.nan, 0) 


# In[15]:


data.WearsGlasses.unique()


# In[16]:


data.HairColor.unique()


# In[17]:


data['HairColor'] = data['HairColor'].replace('0', pd.np.nan) 
data['HairColor'] = data['HairColor'].replace('Unknown', pd.np.nan)


# In[18]:


# Function to implement Target Encoding
def calc_smooth_mean(df, by, on, m):
    mean = df[on].mean()                                 # Compute the global mean
    agg = df.groupby(by)[on].agg(['count', 'mean'])      # Compute number of values & mean of each group
    counts = agg['count']
    means = agg['mean']

    smooth = (counts * means + m * mean) / (counts + m)  # Compute the "smoothed" means
    # Replace each value by the according smoothed mean
    return df[by].map(smooth)


# In[19]:


# Target Encoding on Country
data['Country'] = calc_smooth_mean(data, 'Country', 'Income', 2)


# In[20]:


# Target Encoding on Profession (More weight given for column with null values)
data['Profession'] = calc_smooth_mean(data, 'Profession', 'Income', 50)


# In[21]:


data.head()


# In[22]:


data.isnull().sum()


# In[23]:


data.head()


# In[24]:


'''
data.drop("YearOfRecord", axis=1)
data.drop("Country", axis=1)
data.drop("WearsGlasses", axis=1)
'''


# In[25]:


# One Hot Encoding 
data1 = pd.get_dummies(data, columns=["Gender"], drop_first = True)
#data1 = pd.get_dummies(data1, columns=["Country"], drop_first = True)
#data1 = pd.get_dummies(data1, columns=["Profession"], drop_first = True)
#data1 = pd.get_dummies(data1, columns=["UniversityDegree"], drop_first = True)
data1 = pd.get_dummies(data1, columns=["HairColor"], drop_first = True)


# In[26]:


pd.set_option('display.max_columns', 100)
data1


# In[27]:


data1.isnull().sum()


# In[28]:


data1.tail()


# In[29]:


X_train = data1[0:len(data_train)]          # Dataframe again split into training and test dataset


# In[30]:


len(X_train)


# In[31]:


# Fill null values 
X_train["YearOfRecord"].fillna((X_train["YearOfRecord"].mean()), inplace=True )
X_train["Age"].fillna((X_train["Age"].mean()), inplace=True )
X_train["Profession"].fillna((X_train["Profession"].mean()), inplace=True )


# In[32]:


X_train.isnull().sum()


# In[33]:


Y_train = X_train[["Income"]]                # Split table into predictors and response
X_train = X_train.drop("Income", axis=1)


# In[34]:


# Split Training Data into training and holout data 

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
X_training, X_holdOut, Y_training, Y_holdOut = train_test_split(X_train, Y_train, train_size=0.9, random_state=100)


# In[35]:


# Apply RandomForestRegressor on training data

from sklearn.ensemble import RandomForestRegressor
LR = RandomForestRegressor(n_estimators=1000, random_state=100)
#LR = RandomForestRegressor(max_depth=4, max_features='log2', min_samples_leaf=0.1, n_estimators=400, random_state=100)


# In[36]:


model = LR.fit(X_training, Y_training)        # Fit the RFR model


# In[38]:


# Predict the holdout data
from sklearn.metrics import mean_squared_error, r2_score
ypred = model.predict(X_holdOut)


# In[39]:


# Calculate the different metrics
import math

mse = mean_squared_error(Y_holdOut, ypred)
rmse = math.sqrt(mse)
rmse


# In[40]:


####################################### Prediction Time####################################

X_test = data1[len(data_train):]
X_test = X_test.drop("Income", axis=1)


# In[ ]:


X_test.isnull().sum()


# In[ ]:


# Fill all null values with mean
X_test["YearOfRecord"].fillna((X_test["YearOfRecord"].mean()), inplace=True )
X_test["Age"].fillna((X_test["Age"].mean()), inplace=True )
X_test["Profession"].fillna((X_test["Profession"].mean()), inplace=True )
X_test["Country"].fillna((X_test["Country"].mean()), inplace=True )


# In[ ]:


X_test.isnull().sum()


# In[ ]:


Y_pred = model.predict(X_test)      # Predict response for out of sample data


# In[ ]:


# Write predicted data to CSV
Y_pred = pd.DataFrame(Y_pred)
Y_pred.to_csv("E:/Trinity/Machine Learning/Kaggle/tcdml1920-income-ind/submission.csv", sep=',', index=False, header=True)

