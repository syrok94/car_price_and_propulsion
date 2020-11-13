#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score


# In[2]:


#loading dataset
data=pd.read_csv(r"C:\Users\bobby patel\Desktop\car_price\propulsion.csv",index_col="Unnamed: 0")


# In[3]:


#viewing top 5 data
data.head()


# # predicting GT Turbine decay state coefficient.

# In[4]:


#viewing statistics ofdata
data.describe()


# In[5]:


#checking for null value
data.isnull().any()


# In[6]:


#assigning feature and target variable to x and y
x=data.drop("GT Turbine decay state coefficient.",axis=1)
y=data.get("GT Turbine decay state coefficient.")


# In[7]:


#plotting heatmap to check corelation
corelation=x.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corelation,annot=True,)


# - since,all the features are highly co-related to each other we need to select best features to evalute models.

# ### now we use f-test for feature selection and first select 5,10,and 15 best features and evaluate our model

# In[8]:


#using scikit learn to find f-test
from sklearn.feature_selection import f_regression
def f_test(feature,target):
  f_test=f_regression(feature,target)
  f_values=f_test[0]
  f_values=pd.Series(f_values,index=feature.columns)
  return f_values


# In[9]:


#calling f_test for x
values=f_test(x,y)


# In[10]:



#assigning top 5,10 and 15 best features
top5_f_value=values.nlargest(5)
top10_f_value=values.nlargest(10)
top15_f_value=values.nlargest(15)


# In[11]:


#defining a function for Evaluating model
def Model_Evaluation(best_features):
  X=data.loc[:,best_features.index]
  Y=y
  lr_reg=LinearRegression()
  dt_reg=DecisionTreeRegressor()
  rf_reg=RandomForestRegressor()
  svr_reg=SVR()
  model=[lr_reg,dt_reg,rf_reg,svr_reg]
  for i in model:
    evaluate=cross_val_score(i,X,Y,cv=10,scoring="neg_mean_squared_error",verbose=1).mean()
    rmse_val=-evaluate
    rmse=print("rmse of "+str(i)+" is={}".format(rmse_val))

  return rmse


# In[12]:


#modl evaluation using 15 best feature
Model_Evaluation(top15_f_value)


# In[13]:


#model evaluation using 10 best features
Model_Evaluation(top10_f_value)


# In[14]:


#model evaluation using 5 best features
Model_Evaluation(top5_f_value)


# - from above results we conclude that Random Forest Regressor with 15 best feature has lowest **rmse** value

# #### now we use Extra Tree Regressor to select features

# In[15]:


from sklearn.tree import ExtraTreeRegressor
ex=ExtraTreeRegressor()
ex.fit(x,y)


# In[16]:


imp_feature=ex.feature_importances_
imp_feature=pd.Series(imp_feature,index=x.columns)


# In[17]:


top5_imp=imp_feature.nlargest(5)
top10_imp=imp_feature.nlargest(10)
top15_imp=imp_feature.nlargest(15)


# In[18]:


Model_Evaluation(top5_imp)


# In[19]:


Model_Evaluation(top10_imp)


# In[20]:


Model_Evaluation(top15_imp)


# - from above results again RandomForest Regressor gives lowest **rmse** when using features selected from ExtraTreeRegressor.

# #### so, i am going to use 15 features selected from f-test

# In[21]:


#assigning X and Y 
X=data.loc[:,top15_f_value.index]
Y=y
#splitting X andY
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=42)
#defining model 
model=RandomForestRegressor()
#fitting model
model.fit(X_train,Y_train)
#model prediction
model_pred=model.predict(X_test)


# In[22]:


#finding accuracy of model
from sklearn.metrics import r2_score
print("accuracy of model :{}".format(r2_score(model_pred,Y_test)))


# In[23]:


#plotting prediction and true value
plt.figure(figsize=(7,5))
ax=sns.distplot(model_pred,hist=False,label="pred")
sns.distplot(Y_test,hist=False,label="True",ax=ax)


# # predicting GT Compressor decay state coefficient. 

# In[24]:


x_2=data.drop("GT Compressor decay state coefficient.",axis=1)
y_2=data.get("GT Compressor decay state coefficient.")


# In[25]:


corelation_2=x_2.corr()


# In[26]:


plt.figure(figsize=(15,15))
sns.heatmap(corelation_2,annot=True)


# In[27]:


#finding best features for GT Compressor decay state coefficient.
values_2=f_test(x_2,y_2)


# In[28]:


#defining variables to top 5,10 and 15 best features
top5_f_value2=values_2.nlargest(5)
top10_f_value2=values_2.nlargest(10)
top15_f_value2=values_2.nlargest(15)


# In[29]:


#evaluating model
Model_Evaluation(top5_f_value2)


# In[30]:


Model_Evaluation(top10_f_value2)


# In[31]:


Model_Evaluation(top15_f_value2)


# ### again with 15 best feature **rmse** is minimum for RandomForestRegressor

# In[32]:


#splitting data to train and test
x2_train,x2_test,y2_train,y2_test=train_test_split(x_2,y_2,test_size=0.2,shuffle=True,random_state=42)


# In[33]:


#fitting models
model2=RandomForestRegressor()
model2.fit(x2_train,y2_train)
model2_prediction=model2.predict(x2_test)


# In[34]:


#model accuracy
print("model accuracy:{}".format(r2_score(model2_prediction,y2_test)))


# In[35]:


#plotting prediction and true value
plt.figure(figsize=(7,5))
ax=sns.distplot(model2_prediction,hist=False,label="pred")
sns.distplot(y2_test,hist=False,label="true")


# In[ ]:




