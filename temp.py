# Get the dataset --> Loan Prediction

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
train = pd.read_csv("train.csv")
train.head()
test = pd.read_csv("test.csv")
test.head()

train_original = train.copy()
test_original = test.copy()

# Preparing training and testing dataset
x_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

# Missing value treatment
x_train.isnull().sum()
x_train.info()
""" Missing columns are present there. We will treat missing values in all features one by one."""

#For numerical variables
from sklearn.impute import SimpleImputer
imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit(x_train.loc[:,['Outlet_Size']])
x_train.loc[:,['Outlet_Size']] = imputer_mode.transform(x_train.loc[:,['Outlet_Size']])

imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(x_train.loc[:,['Item_Weight']])
x_train.loc[:,['Item_Weight']] = imputer_mean.transform(x_train.loc[:,['Item_Weight']])

#For categorical variables - treatment
x_train['Item_Fat_Content'] = pd.Series(['Low_Fat' if x[0].lower() == 'l' else 'Regular' for x in x_train['Item_Fat_Content']])

# Outlier treatment -- NA

"""Since, machine learning models are based on Mathematical equations and 
you can intuitively understand that it would cause some problem 
if we can keep the Categorical data in the equations because we would only want numbers in the equations.
"""
x_train_temp = pd.get_dummies(x_train['Item_Type'])
x_train = pd.concat([x_train,x_train_temp], axis=1)
x_train = x_train.drop(['Item_Type'], axis=1)

x_train_temp = pd.get_dummies(x_train['Outlet_Type'])
x_train = pd.concat([x_train_temp,x_train], axis=1)
x_train = x_train.drop(['Outlet_Type'], axis=1)

x_train_temp = pd.get_dummies(x_train['Outlet_Location_Type'])
x_train = pd.concat([x_train_temp,x_train], axis=1)
x_train = x_train.drop(['Outlet_Location_Type'], axis=1)

outlet_size = {
    'Small': 1,
    'Medium': 2,
    'High': 3
}
x_train['Outlet_Size'] = x_train['Outlet_Size'].apply(lambda x: outlet_size[x])

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
x_train['Item_Fat_Content'] = ordinal_encoder.fit_transform(x_train.loc[:,['Item_Fat_Content']])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
x_train_stan = x_train.drop(x_train.loc[:,['Outlet_Identifier', 'Item_Identifier']], axis=1)
#x_train_stan = standard_scaler.fit_transform(x_train_stan)

#Check for linear relationship
#Check for multi-collinearity

import seaborn as sns
sns.heatmap(x_train.corr())
# can see all the numerical features are highly correlated --> dataset is wrong



# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train.loc[:,['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']] = scaler.fit_transform(x_train.loc[:,['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']])















from sklearn.linear_model import Ridge
RR = Ridge(alpha=0.05,normalize=True)
modelfit(RR, train_df, test_df, predictors, target, IDcol, 'RR.csv')

from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(DT, train_df, test_df, predictors, target, IDcol, 'DT.csv')

RF = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(RF, train_df, test_df, predictors, target, IDcol, 'RF.csv')



from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, 
             eval_set=[(test_df[predictors], test_df[target])], verbose=False)

#Predict training set:
train_df_predictions = my_model.predict(train_df[predictors])
# make predictions
predictions = my_model.predict(test_df[predictors])

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))
IDcol.append(target)
submission = pd.DataFrame({ x: test_df[x] for x in IDcol})
submission.to_csv("merda.csv", index=False)
















