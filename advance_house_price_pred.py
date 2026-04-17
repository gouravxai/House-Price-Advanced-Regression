import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict , cross_val_score , cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , root_mean_squared_error
import xgboost as xgb
import joblib
df = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\data-science-projects\house price advanced regression\train.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace=True)
print(df.isnull().sum())
df.fillna(df.median(numeric_only=True),inplace = True)
df.replace([np.inf,-np.inf],0,inplace=True)
print(df.isnull().sum().sum())
df.drop(columns=['Id','Alley','PoolQC','Fence','MiscFeature','LowQualFinSF','3SsnPorch','ScreenPorch'], inplace=True)
selected_cols = [
'OverallQual','OverallCond','GrLivArea','TotalBsmtSF','1stFlrSF','2ndFlrSF',
'FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd',
'GarageCars','GarageArea',
'YearBuilt','YearRemodAdd','YrSold',
'Neighborhood','MSZoning',
'LotArea','LotFrontage','MasVnrArea','BsmtFinSF1','OpenPorchSF',
'SalePrice'
]

df = df[selected_cols]
# ploting Q-Q plot for Saleprice and GrLivArea
stats.probplot(df['SalePrice'],dist='norm',plot=plt)
plt.title("Q-Q Plot before LOG")
plt.show()
df['SalePrice'] = np.log1p(df['SalePrice'])
stats.probplot(df['SalePrice'],dist='norm',plot=plt)
plt.title('Q-Q plot after LOG')
plt.show()
stats.probplot(df['GrLivArea'],dist='norm',plot=plt)
plt.show()
df['GrLivArea'] = np.log1p(df['GrLivArea'])
stats.probplot(df['GrLivArea'],dist='norm',plot=plt)
plt.show()
df['LotArea'] = np.log1p(df['LotArea'])
df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath'])
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
df['RoomsPerArea'] = df['TotRmsAbvGrd'] / df['GrLivArea']
df['BasementRatio'] = df['TotalBsmtSF'] / df['GrLivArea']
df = pd.get_dummies(df, columns=['Neighborhood','MSZoning'], drop_first=True)
x = df.drop('SalePrice',axis=1)
y=df['SalePrice']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=42)
lr = LinearRegression()
lr.fit(x_train,y_train)
prediction1 = lr.predict(x_test)
print('Performance of Linear Regression model : ',root_mean_squared_error(y_test,prediction1))
rf = RandomForestRegressor(n_estimators = 100,random_state=42)
rf.fit(x_train,y_train)
prediction2 = rf.predict(x_test)
rmse_rf = np.sqrt(mean_squared_error(y_test,prediction2))
print('Random Forest RMSE : ',rmse_rf)
xgb_model = xgb.XGBRegressor(n_estimators = 1000,learning_rate = 0.05,max_depth=4,random_state=42,subsample = 0.8,colsample_bytree=0.8)
xgb_model.fit(x_train,y_train)
prediction3 = xgb_model.predict(x_test)
rmse_xgb = root_mean_squared_error(y_test,prediction3)
print('xgboost rmse :',rmse_xgb)
scores = cross_val_score(lr,x,y,cv=5,scoring = 'neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print('RMSE scores :',rmse_scores)
print('Average Means : ',rmse_scores.mean())
importances = rf.feature_importances_
feature_names = x.columns
feature_importances_df = pd.DataFrame({'feature' : feature_names,'importance' : importances})
feature_importances_df = feature_importances_df.sort_values(by='importance',ascending=False)
print(feature_importances_df.head(10))
joblib.dump(lr, 'house_price_model.pkl')
joblib.dump(list(x.columns), 'model_columns.pkl')