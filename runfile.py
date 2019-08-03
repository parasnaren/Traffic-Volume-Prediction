import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr
from functools import reduce
from sklearn.metrics import mean_squared_error
from math import sqrt


"""
Preprocessing functions
"""
def rmse(y_actual, y_pred):
    return sqrt(mean_squared_error(y_actual, y_pred))

 
def plotVsGraph(df, a, b):
    plt.scatter(df[a], df[b])

def printStatistic(df, col):
    print('Unique value: ', df[col].unique())
    print('Value counts: ', df[col].value_counts())
    
def getHeatmap(data):
    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );

"""
Visualisations and Analysis
"""

df = pd.read_csv('data/Train.csv')
df.isnull().sum()
df.dtypes
getHeatmap(df)

test = pd.read_csv('data/Test.csv')

for col in list(df.columns):
    print(col, '\t', df[col].nunique())
    

# Holidays categorical
holidays = pd.get_dummies(df['is_holiday']).iloc[:, 1:]
t_holidays = pd.get_dummies(test['is_holiday']).iloc[:, 1:]

# Snow analysis (only binary)
df['snow_p_h'].value_counts()
snowbins = np.linspace(0,80,20)
d = df[df.snow_p_h == 0]
dd = df[df.snow_p_h != 0]
plt.hist(df[df.snow_p_h == 0].traffic_volume, snowbins, alpha=0.5, label='snow')
plt.hist(df[df.snow_p_h != 0].traffic_volume, snowbins, alpha=0.5, label='no snow')
snow = df['snow_p_h'].apply(lambda x: 1 if x > 0 else 0)
t_snow = test['snow_p_h'].apply(lambda x: 1 if x > 0 else 0)

# Date-time analysis
import datetime 
import calendar 
  
def findDay(date):
    born = datetime.datetime.strptime(date, '%Y %m %d').weekday()
    return (calendar.day_name[born])

    # get dates
date = df['date_time'].apply(lambda x: x.split(' ')[0].replace('-',' '))
day = date.apply(lambda x: findDay(x))
day = pd.get_dummies(day).iloc[:, 1:]

t_date = test['date_time'].apply(lambda x: x.split(' ')[0].replace('-',' '))
t_day = date.apply(lambda x: findDay(x))
t_day = pd.get_dummies(day).iloc[:, 1:]

    # get time
time = df['date_time'].apply(lambda x: x.split(' ')[1].split(':')[0])
time = pd.get_dummies(time).iloc[:, 1:]

t_time = test['date_time'].apply(lambda x: x.split(' ')[1].split(':')[0])
t_time = pd.get_dummies(time).iloc[:, 1:]


# Rain (only binary)
plt.hist(df['rain_p_h'])
plt.show()
rain = df['rain_p_h'].apply(lambda x: 1 if x > 0 else 0)
t_rain = test['rain_p_h'].apply(lambda x: 1 if x > 0 else 0)

# Visibility (ordinal) -dew redundant same as visibility
plt.hist(df['visibility_in_miles'])
plt.show()
visible = df['visibility_in_miles']
t_visible = test['visibility_in_miles']

# Clouds
plt.hist(df['clouds_all'], bins=5)
plt.show()

def cloud_division(data):
    if(0 <= data < 20):
        return 1
    elif(20 < data <= 40):
        return 2
    elif(40 < data <= 60):
        return 3
    elif(60 < data <= 80):
        return 4
    else:
        return 5
    
cloud = df['clouds_all'].apply(lambda x: cloud_division(x))
t_cloud = test['clouds_all'].apply(lambda x: cloud_division(x))

# Numerical values
plt.hist(df['humidity'])
plt.show()
corr, _ = pearsonr(df['humidity'], df['traffic_volume'])
print('Correlation with humidity %.3f' % corr)

plt.hist(df['wind_speed'])
corr, _ = pearsonr(df['wind_speed'], df['traffic_volume'])
print('Correlation with wind speed %.3f' % corr)

plt.hist(df['wind_direction'])
plt.show()
corr, _ = pearsonr(df['wind_direction'], df['traffic_volume'])
print('Correlation with wind direction %.3f' % corr)


"""
Baseline
"""
df = pd.read_csv('Train.csv')
y = df['traffic_volume'].values

df.columns

# humidity and wind speed added
X = df.iloc[:, [3,4]]
X_test = test.iloc[:, [3,4]]


cols = [pd.DataFrame(x) for x in [day, time, snow, rain, visible, 
        cloud, holidays, X]]
X = reduce(lambda left, right: pd.merge(left, right, 
                                        left_index=True,
                                        right_index=True), cols)


t_cols = [pd.DataFrame(x) for x in [t_day, t_time, t_snow, t_rain, 
          t_visible, t_cloud, t_holidays, X_test]]
X_test = reduce(lambda left, right: pd.merge(left, right, 
                                        left_index=True,
                                        right_index=True), t_cols)

missing_cols = set(X.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0
X_test = X_test[X.columns]


"""
Testing
"""
X_train, X_val, y_train, y_val = tts(X, y, test_size=0.25)

forest = RandomForestRegressor(n_estimators=100)

forest.fit(X_train, y_train)
y_pred = forest.predict(X_val)

print("RMSE value = %.3f" % rmse(y_val, y_pred))

predictions = forest.predict(X_test)
sample = pd.DataFrame({
        'date_time': test['date_time'],
        'traffic_volume': predictions})
    
sample.to_csv('sub-1.csv', index=False)













