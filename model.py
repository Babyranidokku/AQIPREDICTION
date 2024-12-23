import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



data = pd.read_csv(r'C:\Users\babyr\OneDrive\Desktop\aqiprediction\main\air quality data.csv')




print(data['AQI_Bucket'].value_counts())

print(data.dtypes)

# Convert relevant columns to numeric, forcing errors to NaN

numeric_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

data.head()

data.shape

data.info()

data.isnull().sum()

data.duplicated().sum()

df= data.dropna(subset=['AQI'],inplace=True)

data.isnull().sum().sort_values(ascending=False)

data.shape

data.describe().T

null_values_percentage = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)

null_values_percentage

data['Xylene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['PM10'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['NH3'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['Toluene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['Benzene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['NOx'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['O3'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['PM2.5'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['SO2'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['CO'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['NO2'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['NO'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

data['AQI'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

sns.displot(data, x="AQI", color="purple")
plt.show()

sns.set(style="darkgrid")
graph=sns.catplot(x="City",kind="count",data=data,height=5,aspect=3)
graph.set_xticklabels(rotation=90)

sns.set(style="darkgrid")
graph=sns.catplot(x="City",kind="count",data=data,col="AQI_Bucket",col_wrap=2,height=3.5,aspect=3)
graph.set_xticklabels(rotation=90)

graph1=sns.catplot(x="City",y="PM2.5",kind="box",data=data,height=5,aspect=3)
graph1.set_xticklabels(rotation=90)

graph2=sns.catplot(x="City",y="NO2",kind="box",data=data,height=5,aspect=3)
graph2.set_xticklabels(rotation=90)

graph3=sns.catplot(x="City",y="O3",data=data,kind="box",height=5,aspect=3)
graph3.set_xticklabels(rotation=90)

graph4=sns.catplot(x="City",y="SO2",data=data,kind="box",height=5,aspect=3)
graph4.set_xticklabels(rotation=90)

graph5=sns.catplot(data=data,kind="box",x="City",y="NOx",height=6,aspect=3)
graph5.set_xticklabels(rotation=90)

graph6=sns.catplot(data=data,kind="box",x="City",y="NO",height=6,aspect=3)
graph6.set_xticklabels(rotation=90)

graph7=sns.catplot(x="AQI_Bucket",data=data,kind="count",height=6,aspect=3)
graph7.set_xticklabels(rotation=90)



data.isnull().sum().sort_values(ascending=False)



data.describe().loc["mean"]

data = data.replace({

"PM2.5" : {np.nan:67.476613},
"PM10" :{np.nan:118.454435},
"NO": {np.nan:17.622421},
"NO2": {np.nan:28.978391},
"NOx": {np.nan:32.289012},
"NH3": {np.nan:23.848366},
"CO":  {np.nan:2.345267},
"SO2": {np.nan:34.912885},
"O3": {np.nan:38.320547},
"Benzene": {np.nan:3.458668},
"Toluene": {np.nan:9.525714},
"Xylene": {np.nan:3.588683}})


data.isnull().sum()

graph=sns.catplot(x="AQI_Bucket",data=data,kind="count",height=6,aspect=3)
graph.set_xticklabels(rotation=90)

data = data.drop(["AQI_Bucket"], axis=1)

data.head()

sns.boxplot(data=data[[ 'PM2.5', 'PM10']])

sns.boxplot(data=data[[ 'NO', 'NO2', 'NOx','NH3']])

sns.boxplot(data=data[[ 'O3', 'CO', 'SO2']])

sns.boxplot(data=data[[ 'Benzene', 'Toluene', 'Xylene']])


def replace_outliers_with_quartiles(df):

    for column in data.select_dtypes(include=['number']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data[column] = data[column].apply(
            lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x)
        )

    return data

data = replace_outliers_with_quartiles(data)

data.describe().T

sns.boxplot(data=data[[ 'PM2.5', 'PM10']])

sns.boxplot(data=data[[ 'NO', 'NO2', 'NOx','NH3']])

sns.boxplot(data=data[[ 'O3', 'CO', 'SO2']])

sns.boxplot(data=data[[ 'Benzene', 'Toluene', 'Xylene']])

# distribution of aqi from 2015-2020
sns.displot(data, x="AQI", color="red")
plt.show()

df1=data.drop(columns=['City'])

numeric_df = df1.select_dtypes(include=['number'])
print(df1.dtypes)
df1['AQI'] = pd.to_numeric(df1['AQI'], errors='coerce')


plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()


data.head()

data


data.drop(['Date'],axis=1,inplace=True)
data.drop(['City'],axis=1,inplace=True)

data

from sklearn.preprocessing import StandardScaler
df1 = StandardScaler().fit_transform(data)

df1

data = pd.DataFrame(df1,columns = data.columns)

data

if 'AQI_Bucket' in data.columns:
    encoder = LabelEncoder()
    data['AQI_Bucket'] = encoder.fit_transform(data['AQI_Bucket'])

# Select numeric columns
data_numeric = data[numeric_columns]

# Scale only numeric columns
scaler = StandardScaler()
# Scale numeric columns only
data_scaled = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)


# Combine scaled data with non-numeric columns if required
data_scaled_full = data.copy()
data_scaled_full[numeric_columns] = data_scaled

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Data Preparation for Modeling
x=data[["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]]
y=data["AQI"]

x.head()

y.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)  # Apply to features only


# Ensure correct bucket mapping
bins = [0, 50, 100, 200, 300, 400, 500]
labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
data['AQI_Bucket'] = pd.cut(data['AQI'], bins=bins, labels=labels)


import numpy as np

data['AQI_transformed'] = np.log1p(data['AQI'])


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=70)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# splitting the data into training and testing data

model=LinearRegression()
model.fit(X_train,Y_train)

#predicting train
train_pred=model.predict(X_train)
#predicting on test
test_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score # Already imported
import sklearn.metrics as metrics 


RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_pred)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_pred)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',model.score(X_train, Y_train))
print('RSquared value on test:',model.score(X_test, Y_test))

KNN = KNeighborsRegressor()
KNN.fit(X_train,Y_train)

#predicting train
train_pred=model.predict(X_train)
#predicting on test
test_pred=model.predict(X_test)

RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_pred)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_pred)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',model.score(X_train, Y_train))
print('RSquared value on test:',model.score(X_test, Y_test))

DT=DecisionTreeRegressor()
DT.fit(X_train,Y_train)

#predicting train
train_preds=DT.predict(X_train)
#predicting on test
test_preds=DT.predict(X_test)

RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',DT.score(X_train, Y_train))
print('RSquared value on test:',DT.score(X_test, Y_test))

RF=RandomForestRegressor(n_estimators=100, random_state=42)
RF.fit(X_train,Y_train)

#predicting train
train_preds1=RF.predict(X_train)
#predicting on test
test_preds1=RF.predict(X_test)

RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds1)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds1)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',RF.score(X_train, Y_train))
print('RSquared value on test:',RF.score(X_test, Y_test))



from sklearn.preprocessing import StandardScaler
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

scaler = StandardScaler()
scaler.fit(X_train)  # Use your training data features here

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)



# Test data
test_data = pd.DataFrame({
    'PM2.5': [50, 75],
    'PM10': [120, 160],
    'NO': [20, 25],
    'NO2': [30, 40],
    'NOx': [45, 60],
    'NH3': [10, 15],
    'CO': [1.5, 2.0],
    'SO2': [5, 7],
    'O3': [20, 25],
    'Benzene': [2, 3],
    'Toluene': [3, 4],
    'Xylene': [0.5, 0.6]
})

# Ensure columns are in the expected order
expected_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
test_data = test_data[expected_features]

# Scale the test data using the same scaler used in training
test_data_scaled = scaler.transform(test_data)

# Predict AQI
predictions = model.predict(test_data_scaled)

# Map predictions to AQI buckets
def map_aqi_to_bucket(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

# Map predictions
aqi_buckets = [map_aqi_to_bucket(aqi) for aqi in predictions]

# Output results
print("Predicted AQI values:", predictions)
print("AQI Categories:", aqi_buckets)
