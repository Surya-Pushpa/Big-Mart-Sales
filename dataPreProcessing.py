#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing Libraries
import numpy as np
import pandas as pd
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#Importing dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_original = train.copy()
test_original = test.copy()

# Join Train and Test Dataset
#Create source column to later separate the data easily
train['source']='train'
test['source']='test'
data = pd.concat([train,test], ignore_index = True)
print(train.shape, test.shape, data.shape)

# Data Exploring
data.info()
data.isnull().sum()
sns.heatmap(data.corr())
"""
Item_Identifier                 0 -> C -> 1559/8523 unique
Item_Weight                  1463 -> N -> Missing Values -> Low correlation
Item_Fat_Content                0 -> C -> 2/5 unique -> low have high sales
Item_Visibility                 0 -> N -> high - low sales, low -> high sales
Item_Type                       0 -> C -> 16 high unique
Item_MRP                        0 -> N -> correlated > 0.5
Outlet_Identifier               0 -> C ->
Outlet_Establishment_Year       0 -> N ->
Outlet_Size                  2410 -> C -> Missing Values -> mostly medium
Outlet_Location_Type            0 -> C -> can merge but before see impact on sales
Outlet_Type                     0 -> C ->
Item_Outlet_Sales               0 -> N ->
"""

"""========================================== Missing values Treatment========================================== """
#imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit(data.loc[:,['Outlet_Size']])
#data.loc[:,['Outlet_Size']] = imputer_mode.transform(data.loc[:,['Outlet_Size']])

"""Outlet_Size"""
outlet_type_size = data.pivot_table(index='Outlet_Type', values='Outlet_Size', aggfunc=lambda x:x.mode())
def impute_outletSize(cols):
    outletType=cols[0]
    outletSize=cols[1]
    if pd.isnull(outletSize):
        return outlet_type_size[outlet_type_size.index == outletType]['Outlet_Size'][0]
    else:
        return outletSize
data['Outlet_Size'] = data[['Outlet_Type', 'Outlet_Size']].apply(impute_outletSize, axis=1)


"""Item_Weight"""
identifier_weight = data.pivot_table(index='Item_Identifier', values="Item_Weight")
def impute_weight(cols):
    Weight = cols[0]
    Identifier = cols[1]
    if pd.isnull(Weight):
        if identifier_weight['Item_Weight'][identifier_weight.index == Identifier].shape[0] == 0:
            return data['Item_Weight'].mean()
        else:
            return identifier_weight['Item_Weight'][identifier_weight.index == Identifier][0]
    else:
        return Weight
data['Item_Weight'] = data[['Item_Weight','Item_Identifier']].apply(impute_weight,axis=1)

"""Item_Visibility"""
identifier_visibility = data.pivot_table(index='Item_Identifier', values='Item_Visibility', aggfunc=np.mean)
def impute_visibility(cols):
    identifier = cols[0]
    visibility = cols[1]
    
    if visibility == 0:
        return identifier_visibility[identifier_visibility.index == identifier]['Item_Visibility'][0]
    else:
        return visibility
data['Item_Visibility'] = data[['Item_Identifier', 'Item_Visibility']].apply(impute_visibility, axis=1)

"""Item_Visibility_MeanRatio"""
def impute_visibility_ratio(x):
    identifier = x['Item_Identifier']
    visibility = x['Item_Visibility']
    denom = identifier_visibility['Item_Visibility'][identifier_visibility.index == identifier][0]
    return visibility/denom
data['Item_Visibility_MeanRatio'] = data.apply(impute_visibility_ratio, axis=1)




""" =========================================Duplicate values treatment============================================ """

"""========================================Feature Engineering======================================================"""

data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

data['Item_Fat_Content'] = pd.Series(['Low_Fat' if x[0].lower() == 'l' else 'Regular' for x in data['Item_Fat_Content']])
data['Item_Fat_Content'].value_counts()

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"


""" =========================================Impact on sales ======================================================"""
data.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.sum).plot(kind='bar', color='blue',figsize=(12,7))
data.pivot_table(index='Item_Type', values="Item_Outlet_Sales", aggfunc=np.mean).plot(kind='bar', color='blue',figsize=(12,7))
data.pivot_table(index='Outlet_Size', values="Item_Outlet_Sales", aggfunc=np.median).plot(kind='bar', color='blue',figsize=(12,7))
data.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median).plot(kind='bar', color='blue',figsize=(12,7))
data.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median).plot(kind='bar', color='blue',figsize=(12,7))
data.pivot_table(index='Item_Type', values="Item_MRP", aggfunc=np.sum).plot(kind='bar', color='blue',figsize=(12,7))

plt.scatter(data['Item_Weight'], data['Item_Outlet_Sales'])
plt.plot(data.Item_MRP, data["Item_Outlet_Sales"],'.', alpha = 0.3)

plt.scatter(data['Item_Visibility'], data['Item_Outlet_Sales'])
plt.scatter(data['Item_Visibility_MeanRatio'], data['Item_Outlet_Sales'])

data.pivot_table(index='Item_Identifier', values='Item_Visibility', aggfunc=np.mean).plot(kind='bar', figsize=(12,7))

"""==============================================Categorical Hot Encoding==============================================="""

#Import library:

le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i] = le.fit_transform(data[i])

"""==============================================Exporting Data========================================================"""

#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)
#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


"""======================================================Model Building================================================="""
# Importing Libraries
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

#Importing dataset
train_df = pd.read_csv('train_modified.csv')
test_df = pd.read_csv('test_modified.csv')

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    #Remember the target had been normalized
#    Sq_train = (dtrain[target])**2
    Sq_train = (dtrain[target])
    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors],Sq_train , cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(Sq_train.values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    print("R2 Square : " + str(metrics.r2_score(dtrain_predictions.reshape(-1,1),dtrain[target])))
    print("Mean Absolute Error : " + str(mean_absolute_error(Sq_train.values, dtrain_predictions)))
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)



#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
predictors = train_df.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])

"""================================================= Regression Model============================================="""

#from sklearn.linear_model import LinearRegression
#LR = LinearRegression(normalize=False)
#modelfit(LR, train_df, test_df, predictors, target, IDcol, 'LR.csv')
#
#
#from sklearn.linear_model import Ridge
#RR = Ridge(alpha=0.05,normalize=True)
#modelfit(RR, train_df, test_df, predictors, target, IDcol, 'RR.csv')

#from sklearn.tree import DecisionTreeRegressor
#DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
#modelfit(DT, train_df, test_df, predictors, target, IDcol, 'DT.csv')
#
#RF = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
#modelfit(RF, train_df, test_df, predictors, target, IDcol, 'RF.csv')

#
#from xgboost import XGBRegressor
#my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, 
#             eval_set=[(test_df[predictors], test_df[target])], verbose=False)
#
##Predict training set:
#train_df_predictions = my_model.predict(train_df[predictors])
## make predictions
#predictions = my_model.predict(test_df[predictors])
#
#print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
#print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))
#IDcol.append(target)
#submission = pd.DataFrame({ x: test_df[x] for x in IDcol})
#submission.to_csv("xgboost.csv", index=False)
#
#from sklearn.tree import DecisionTreeRegressor
#predictors = [x for x in train_df.columns if x not in [target]+IDcol]
#alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
#modelfit(alg3, train_df, test_df, predictors, target, IDcol, 'alg3.csv')
#coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
#coef3.plot(kind='bar', title='Feature Importances')
#
#predictors = ['Item_MRP','Outlet_Type','Outlet','Outlet_Years']
#alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
#modelfit(alg4, train_df, test_df, predictors, target, IDcol, 'alg4.csv')
#coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
#coef4.plot(kind='bar', title='Feature Importances')
#
#from sklearn.ensemble import RandomForestRegressor
#predictors = [x for x in train_df.columns if x not in [target]+IDcol]
#alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
#modelfit(alg5, train_df, test_df, predictors, target, IDcol, 'alg5.csv')
#coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
#coef5.plot(kind='bar', title='Feature Importances')

from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train_df.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train_df, test_df, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')


from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=2000, learning_rate=0.03)
my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, 
             eval_set=[(test_df[predictors], test_df[target])], verbose=False)
train_df_predictions = my_model.predict(train_df[predictors])
predictions = my_model.predict(test_df[predictors])
print("R2 Square : " + str(metrics.r2_score(train_df_predictions.reshape(-1,1),train_df[target])))
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))
cv_score = cross_val_score(my_model, train_df_predictions.reshape(-1,1),train_df[target] , cv=20, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))


IDcol.append(target)
submission = pd.DataFrame({ x: test_df[x] for x in IDcol})
submission.to_csv("xgboost.csv", index=False)
coef7 = pd.Series(my_model.feature_importances_, predictors).sort_values(ascending=False)
coef7.plot(kind='bar', title='Feature Importances')

