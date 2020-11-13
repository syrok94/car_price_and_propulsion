#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#loading dataset
data=pd.read_csv(r"C:\Users\bobby patel\Desktop\car_price\cars_price.csv",index_col="Unnamed: 0")


# In[3]:


#top 5 data points
data.head()


# In[4]:


#data information 
data.info()


# In[5]:


# it describe stats of data
data.describe()


# In[11]:


#plotting histograms for priceUsd,milage and volume
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.hist(data["priceUSD"],bins=50)
plt.subplot(1,3,2)
plt.hist(data["mileage(kilometers)"],bins=50)
plt.subplot(1,3,3)
plt.hist(data["volume(cm3)"],bins=50)


# - histogram shows data point are right skewed.
# - we use log-transformation and Standard Scaler technique to make data point normally distributed but first we check null values and try to remove them.

# In[12]:


#now,we check data for null values.
#and is found, we take steps to either drop them or impute them
#with some values.
data.isnull().sum()


# In[13]:


from sklearn.impute import SimpleImputer
#imputing null values to volume column
#since,volume is float type data that's why strategy="mean" is used.
SI=SimpleImputer(strategy="mean")
data["volume(cm3)"]=SI.fit_transform(data[["volume(cm3)"]])
#since,segment and drive_unit were object type
#that is why strategy="most frquent" is used.
SI=SimpleImputer(strategy="most_frequent")
data["segment"]=SI.fit_transform(data[["segment"]])
SI=SimpleImputer(strategy="most_frequent")
data["drive_unit"]=SI.fit_transform(data[["drive_unit"]])


# In[14]:


#again,check for null values
data.isnull().any()


# - now, there is no null value in our data.

# In[15]:


data.head()


# In[16]:


#we remove year column and replace it with 
# difference of current year and year which will show how old car is.
data["current_year"]=2020
data["YEAR"]=data["current_year"]-data["year"]
data=data.drop(["year","current_year"],axis=1)


# In[17]:


#plotting priceUSD vs YEAR
sns.scatterplot(data["YEAR"],data["priceUSD"])


# - **YEAR VS priceUSD** plots shows as age of car increase its price decrease.

# In[18]:


data.head()


# In[19]:


#we see how volume and mileage are related to priceUSD
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(data["mileage(kilometers)"],data["priceUSD"],marker="X",color="red")
plt.subplot(1,2,2)
sns.regplot(data["volume(cm3)"],data["priceUSD"],marker="X",color="red")


# - plot shows there are some outlier in dataset.
# - car with less mileage has higher price.
# - from plot we see volume and price are linearly co-related.
# 

# In[20]:


# here we group categorical data and sort then in descending order of priceUSD
#defining a function grouping_data
def grouping_data(feature):
  group=data.groupby(data[feature]).mean()
  group_sort=group.sort_values(by="priceUSD",ascending=False)
  sns.barplot(group_sort.priceUSD,group_sort.index)


# In[21]:


#grouing color wise
grouping_data("color")


# In[22]:


#grouping segment wise
grouping_data("segment")


# In[23]:


#grouping transmission wise
grouping_data("transmission")


# In[24]:


#grouping drive_unit
grouping_data("drive_unit")


# In[25]:


#grouping condition wise
grouping_data("condition")


# ## creating one-hot-encoding of categorical data
# 

# In[26]:


#before one-hot-encoding we drop some columns 
data=data.drop(["make","model"],axis=1)
data=pd.get_dummies(data,drop_first=True)


# In[27]:


data.head()


# In[28]:


data.info()


# In[29]:


#we use heatmap to see co-relations between continous data points only..
cont_data=data.loc[:,["mileage(kilometers)","volume(cm3)","YEAR","priceUSD"]]
cm=cont_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True)


# - plot shows YEAR and priceUSD are highly negatevely co-related which is alredy seen earlier.

# In[30]:


#creating independent variable x and dependent varible y
x=data.drop("priceUSD",axis=1)
y=data.get("priceUSD")


# In[31]:


#now we select best feature using f-test and p-value among them
from sklearn.feature_selection import f_regression
f_reg=f_regression(x,y)


# In[32]:


f_reg


# - from f-test value and p-value we see that all the selected features are highly co-related
# - every feature is highly co-related but we use another method to be more clarify about co-realations

# In[33]:


#we use ExtraTreeRegressor to get the important feature 
from sklearn.tree import ExtraTreeRegressor
et_reg=ExtraTreeRegressor()
et_reg.fit(x,y)


# In[34]:


#getting list of important features
et_reg.feature_importances_


# In[35]:


#now,we plot the top 10 most important feature
imp_feature=pd.Series(et_reg.feature_importances_,index=x.columns)
imp_feature.nlargest(10).plot(kind="barh")
plt.show()


# In[36]:


#here we perform log-transformation to priceUSD and volume as these data contain outlier and is right skew
# after performing log-transformation now columns are distributed normally
data["volume(cm3)"]=np.log(data["volume(cm3)"])
data["priceUSD"]=np.log(data["priceUSD"])


# In[38]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(data["volume(cm3)"],bins=50)
plt.subplot(1,2,2)
plt.hist(data["priceUSD"],bins=50)


# - we see that now priceUSD and volume are normally distributed.

# In[39]:


# we now define dependent and independent variable to X and Y
X=data.loc[:,['YEAR', 'transmission_mechanics', 'volume(cm3)', 'mileage(kilometers)',
       'drive_unit_front-wheel drive', 'drive_unit_rear drive',
       'fuel_type_petrol', 'fuel_type_electrocar',
       'drive_unit_part-time four-wheel drive', 'color_other']]
Y=data.get("priceUSD")


# In[40]:


#spliting data into test and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=42)


# ## scaling features

# In[41]:


#using standard scaler to standardise values
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)


# ## creating RandomForest model

# In[42]:


#importing randomforestregressor and fit data to it.
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor()
rf_reg.fit(x_train,y_train)



# In[43]:


#now we check how well train and test data are fitted
rf_reg.score(x_train,y_train)
                     
                     


# In[44]:


#now, we predict test data
rf_pred=rf_reg.predict(x_test)


# In[45]:


#here,we check how well test and train set fitted to model.
rf_reg.score(x_test,y_test)


# - we see that the model is highly overfitted.
# - we now perform hyperparameter tunning to get better results

# ## Hyperparameter Tunning

# In[46]:


#import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[47]:


#defining parameters
max_depth=[2,4,5,7,8,10]
max_features=["auto","sqrt"]
max_depth=[int(x) for x in np.linspace(5,30,6)]
min_samples_leaf=[1,2,3,4,5]
n_estimators=[int(x) for x in np.linspace(100,1200,12)]
min_sample_split=[1,2,4,5,10]


# In[48]:


#coverting parameters in form of dict
param={
    "max_depth":max_depth,
    "max_features":max_features,
    "max_depth":max_depth,
    "min_samples_leaf":min_samples_leaf,
    "n_estimators":n_estimators,
    
}


# In[49]:


#Defining RandomizedsearchCV object to perform tunning
random_reg=RandomizedSearchCV(rf_reg,param_distributions=param,n_iter=10,scoring="neg_mean_squared_error",cv=5,verbose=1,random_state=42)


# In[50]:


#fitting x_train and y_train to Randomizedsearchcv model
random_reg.fit(x_train,y_train)


# In[51]:


#we now call best_params_ to get best parameters
random_reg.best_params_


# In[52]:


#here we use best_score_ to get the negative of mean squared error
random_reg.best_score_


# In[53]:


#here,we csall best_estimator_ to get the best estimator model with parameters
# we use this model to get better rsults as describe previously
random_reg.best_estimator_


# In[54]:


#we now use best estimator 
rf_reg=RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=20, max_features='sqrt', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=3,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=1000, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
#fitting best estimator
rf_reg.fit(x_train,y_train)


# In[55]:


#this shows the r^2 score 
#how good x_train and y_train is fitted to best estimator model
rf_reg.score(x_train,y_train)


# In[56]:


#best fitted score of x_test and y_test
rf_reg.score(x_test,y_test)


# - from above r^2 score values we can conclude that now overfitting is reduces

# In[57]:


#prediction from best estimator model
y_pred=rf_reg.predict(x_test)


# In[58]:


#now , se mean_squared_error of best estimator 
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_pred,y_test))


# In[59]:


#here, we plot dist plot to get insight of how good is prediction
ax=sns.distplot(y_pred,hist=False,label="y_pred")
sns.distplot(y_test,hist=False,ax=ax,label="y_test")
plt.legend()


# In[60]:


#here we import pickle and save our model
#import pickle
#file=open("car_price_prediction.pkl","wb")
#pickle.dump(rf_reg,file)


# In[ ]:




