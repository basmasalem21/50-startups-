#%%-------------------------------------------------
#%% importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
#%%----------------------------------------------------
# إعدادات الطباعة
pd.set_option('display.max_columns', None)  # إظهار جميع الأعمدة
pd.set_option('display.max_rows', None)     # إظهار جميع الصفوف (إن لزم الأمر)
pd.set_option('display.float_format', '{:.2f}'.format)  # تقليل الأرقام العشرية (اختياري)
#%%----------------------------------------------------
#%%importing dataset
dataset_path="C:\\Users\\Elhussien\\Desktop\\machine learning\\50_Startups.csv"
dataset=pd.read_csv(dataset_path)
print(dataset.info())
#%%--------------------------------------------
#%% calculating corrolation btw columns
dataset2=dataset.copy()
dataset2=dataset.drop('State',axis=1)
corr=dataset2.corr()
print(corr)
#%%--------------------------------------------
#%%drawing heatmap
sns.heatmap(corr,annot=True)
plt.show()
#%%---------------------------------------------
#%%drawing histogram:-
dataset2.hist(figsize=(13,20))
plt.show()
#%%--------------------------------------------
#%%describing data
describing_data=dataset2.describe().T
print(describing_data)
#%%--------------------------------------------
#%%apply one hot encoding on data
print(dataset['State'].unique())
new_state= pd.get_dummies(dataset['State'],dtype=int,prefix='state',drop_first=True)
#print(new_state)
dataset = pd.concat([dataset, new_state], axis=1)
dataset.drop('State', axis=1, inplace=True)

#print(dataset.head())
print(dataset.info())
#%%----------------------------------------------
#%%determining x and y:-
x=dataset.drop(['Profit'],axis=1)
y=dataset['Profit']
#%%----------------------------------------------
#%%reshaping x and y
#print(x.shape)
#print(y.shape)
y=y.values.reshape(-1,1)
#print(x.shape)
#print(y.shape)
#%%----------------------------------------------
#%%------------------------------------------
#%%drawing first 10 columns with their data
df1 = dataset2.head(10)
df1.plot(kind="bar", figsize=(16,10))
plt.grid(which="major",linestyle="-", linewidth="0.5",color="green")

plt.show()
#%%----------------------------------------------
import statsmodels.api as sm
stmodel1=sm.OLS(y,x).fit()
print(stmodel1.summary())

x=x.drop(['state_Florida'],axis=1)
print(x)

stmodel2=sm.OLS(y,x).fit()
print(stmodel2.summary())


x=x.drop(['state_New York'],axis=1)
print(x)

stmodel3=sm.OLS(y,x).fit()
print(stmodel3.summary())
#8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
#%%splitting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=10)
#%%----------------------------------------------
#%%linear regression model
my_model=LinearRegression()
my_model.fit(x_train,y_train)
#%%----------------------------------------------
#%% prediction using model
y_pred=my_model.predict(x_test)
#%%---------------------------------------------
#%%converting array to series for putting them in the dataframe
y_test = y_test.squeeze()  # تحويل إلى Series إذا كانت DataFrame
y_pred = y_pred.squeeze()  # تحويل إلى Series إذا كانت DataFrame
#%%--------------------------------------------
#%%dataframe for y_test,y_pred,error btw them
dataframe=pd.DataFrame({"True":y_test,
    "predicted":y_pred,
    'error':abs(y_test-y_pred)
})
print(dataframe)
#%%---------------------------------------------
#%%printing mse  and r2 score
mse=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"""
      Mean Square Error is {np.sqrt(mse)}
      R2 Score is {r2}
""")
#%%--------------------------------------------
#%%printing coeficient and intercept of the model
print(f"model intercept is : {my_model.intercept_}")
print(f"model coeficient is : {my_model.coef_}")












































