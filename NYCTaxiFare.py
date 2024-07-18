# %% [markdown]
# # NYC Taxi Fare: Deep Learning
# Brashon Ford
# 
# 
# 

# %% [markdown]
# # Setup
# 
# ## Library import
# We import all the required Python libraries

# %%
pip install utils

# %%
# Data manipulation
import pandas as pd
import numpy as np

# Options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30

# Visualizations
import plotly
import plotly.graph_objs as go
import plotly.offline as ply
plotly.offline.init_notebook_mode(connected=True)



import matplotlib as plt
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error


# %%
def preprocess(df): 
    # remove missing values in the dataframe
    def remove_missing_values(df):
        df = df.dropna()
        return df

# %%
# remove outliers in fare amount
def remove_fare_amount_outliers(df, lower_bound, upper_bound):
    df = df[(df['fare_amount'] > lower_bound) & (df['fare_amount'] <= upper_bound)]
    return df

# %%
def feature_engineer(df):
    # create new columns for year, month, day, day of week and hour
def create_time_features(df):
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour
    df = df.drop(['pickup_datetime'], axis=1)
return df


# %% [markdown]
# 
# # Data import
# We retrieve all the required data for the analysis.

# %%
import os
os.getcwd()

# %%
df = pd.read_csv('/Users/brashonford/new-york-city-taxi-fare-prediction/NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

# %% [markdown]
# # Data processing / Handling Missing Values
# 

# %%
df.head()

# %%
print(df.isnull().sum())
print('')


# %%
df = df.dropna()

print(df.describe())

# %%
import matplotlib.pyplot as plt
df['fare_amount'].hist(bins=500)
plt.xlabel("Fare")
plt.title("Histogram of Fares")
plt.show()

# %%
df = df[(df['fare_amount'] >=0) & (df['fare_amount'] <= 100)]

# %%
df['passenger_count'].hist(bins=6, ec='black')
plt.xlabel("Passenger Count")
plt.title("Histogram of Passenger Count")
plt.show()


# %%
df.loc[df['passenger_count']==0, 'passenger_count'] = 1

# %% [markdown]
# ## Details About The Data
# key: This column seems identical to the pickup_datetime column. It was probably used as an unique identifier in the database it was stored in. We can safely remove this column without any loss of information.
# 
# fare_amount: This is the target variable we are trying to predict, the fare amount paid at the end of the trip.
# 
# pickup_datetime: This column contains information on the pickup date (year, month, day of month), as well as the time (hour, minute, seconds).
# 
# pickup_longitude and pickup_latitude: The longitude and latitude of the pickup location.
# 
# dropoff_longitude and dropoff_latitude: The longitude and latitude of the drop off location.
# 
# passenger_count: The number of passengers.

# %%
# range of longitude for NYC
nyc_min_longitude = -74.05
nyc_max_longitude = -73.75

# range of latitude for NYC
nyc_min_latitude = 40.63
nyc_max_latitude = 40.85

df2 = df.copy(deep=True)

for long in ['pickup_longitude', 'dropoff_longitude']:
    df2 = df2[(df2[long] > nyc_min_longitude) & (df2[long] <
                                                 nyc_max_longitude)]

for lat in ['pickup_latitude', 'dropoff_latitude']:
    df2 = df2[(df2[lat] > nyc_min_latitude) & (df2[lat] < 
                                               nyc_max_latitude)]

# %%
#let's define a new function that will take our DataFrame as an input, and plot the pickup locations on a scatterplot. 

landmarks = {'JFK Airport': (-73.78, 40.643),
             'Laguardia Airport': (-73.87, 40.77),
             'Midtown': (-73.98, 40.76),
             'Lower Manhattan': (-74.00, 40.72),
             'Upper Manhattan': (-73.94, 40.82),
             'Brooklyn': (-73.95, 40.66)}



# %%
import matplotlib.pyplot as plt
%matplotlib inline
def plot_lat_long(df, landmarks, points='Pickup'):
    plt.figure(figsize = (12,12)) # set figure size
    if points == 'pickup':
        plt.plot(list(df.pickup_longitude), list(df.pickup_latitude), 
                 '.', markersize=1)
    else:
        plt.plot(list(df.dropoff_longitude), list(df.dropoff_latitude), 
                 '.', markersize=1)

    for landmark in landmarks:
        plt.plot(landmarks[landmark][0], landmarks[landmark][1], 
                '*', markersize=15, alpha=1, color='r') 
        plt.annotate(landmark, (landmarks[landmark][0]+0.005, 
                     landmarks[landmark][1]+0.005), color='r', 
                     backgroundcolor='w') 
 
    plt.title("{} Locations in NYC Illustrated".format(points))
    plt.grid(None)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show()

# %%
plot_lat_long(df2, landmarks, points='Pickup')


# %%
plot_lat_long(df2, landmarks, points='Drop Off')


# %%
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour

# %%
import numpy as np
df['day_of_week'].plot.hist(bins=np.arange(8)-0.5, ec='black', 
                            ylim=(60000,75000))
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.title('Day of Week Histogram')
plt.show()

# %%
df['hour'].plot.hist(bins=24, ec='black')
plt.title('Pickup Hour Histogram')
plt.xlabel('Hour')
plt.show()

# %%
df.plot.scatter('pickup_longitude', 'pickup_latitude')
plt.show()

# %%
df['passenger_count'].hist(bins=6, ec='black')
plt.xlabel("Passenger Count")
plt.title("Histogram of Passenger Count")
plt.show()

# %%
df.loc[df['passenger_count']==0, 'passenger_count'] = 1

# %%
## RemovinG Outliers:

# range of longitude for NYC
nyc_min_longitude = -74.05
nyc_max_longitude = -73.75

# range of latitude for NYC
nyc_min_latitude = 40.63
nyc_max_latitude = 40.85

# only consider locations within NYC
for long in ['pickup_longitude', 'dropoff_longitude']:
    df = df[(df[long] > nyc_min_longitude) & (df[long] < nyc_max_longitude)]

for lat in ['pickup_latitude', 'dropoff_latitude']:
    df = df[(df[lat] > nyc_min_latitude) & (df[lat] <  nyc_max_latitude)]


# %%
# going to create functions to create a simplier project

def preprocess(df): 
    # remove missing values in the dataframe
    def remove_missing_values(df):
        df = df.dropna()
        return df

    # remove outliers in fare amount
    def remove_fare_amount_outliers(df, lower_bound, upper_bound):
        df = df[(df['fare_amount'] >= lower_bound) & 
                (df['fare_amount'] <= upper_bound)]
        return df

    # replace outliers in passenger count with the mode
    def replace_passenger_count_outliers(df):
        mode = df['passenger_count'].mode()
        df.loc[df['passenger_count'] == 0, 'passenger_count'] = mode
        return df
# remove outliers in latitude and longitude
    def remove_lat_long_outliers(df):
        # range of longitude for NYC
        nyc_min_longitude = -74.05
        nyc_max_longitude = -73.75
        # range of latitude for NYC
        nyc_min_latitude = 40.63
        nyc_max_latitude = 40.85
        # only consider locations within New York City
        for long in ['pickup_longitude', 'dropoff_longitude']:
            df = df[(df[long] > nyc_min_longitude) & 
                    (df[long] < nyc_max_longitude)]
        for lat in ['pickup_latitude', 'dropoff_latitude']:
            df = df[(df[lat] > nyc_min_latitude) & 
                    (df[lat] < nyc_max_latitude)]
        return df

    df = remove_missing_values(df)
    df = remove_fare_amount_outliers(df, lower_bound = 0, 
                                     upper_bound = 100)
    df = replace_passenger_count_outliers(df)
    df = remove_lat_long_outliers(df)
    return df


# %%


# %% [markdown]
# ## Feature Engineering:
# using one's domain knowledge of the problem to create new features for the machine learning algorithm

# %%
print(df.head()['pickup_datetime'])

# %%
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour


# %%
print(df.loc[:5,['pickup_datetime', 'year', 'month', 
                 'day', 'day_of_week', 'hour']])

# %%
df = df.drop(['pickup_datetime'], axis=1)

# %%
def euc_distance(lat1, long1, lat2, long2):
    return(((lat1-lat2)**2 + (long1-long2)**2)**0.5)

# %%
df['distance'] = euc_distance(df['pickup_latitude'],
                              df['pickup_longitude'], 
                              df['dropoff_latitude'],
                              df['dropoff_longitude'])

# %% [markdown]
# ### Hypothesis: 
# trip fare is closely correlated to the distance traveled

# %%
df.plot.scatter('fare_amount', 'distance')
plt.show()

# %%
airports = {'JFK_Airport': (-73.78,40.643),
            'Laguardia_Airport': (-73.87, 40.77),
            'Newark_Airport' : (-74.18, 40.69)}

for airport in airports:
    df['pickup_dist_' + airport] = euc_distance(df['pickup_latitude'],
                                                df['pickup_longitude'], 
                                                airports[airport][1], 
                                                airports[airport][0])
    df['dropoff_dist_' + airport] = euc_distance(df['dropoff_latitude'], 
                                                 df['dropoff_longitude'],
                                                 airports[airport][1], 
                                                 airports[airport][0])

# %%
print(df[['key', 'pickup_longitude', 'pickup_latitude', 
          'dropoff_longitude', 'dropoff_latitude', 
          'pickup_dist_JFK_Airport',
          'dropoff_dist_JFK_Airport']].head())

# %%
df = df.drop(['key'], axis=1)

## drop 'key' because its irrelevant 

# %%
def feature_engineer(df):
    # create new columns for year, month, day, day of week and hour
    def create_time_features(df):
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['hour'] = df['pickup_datetime'].dt.hour
        df = df.drop(['pickup_datetime'], axis=1)
        return df

    # function to calculate euclidean distance
    def euc_distance(lat1, long1, lat2, long2):
        return(((lat1-lat2)**2 + (long1-long2)**2)**0.5)

    # create new column for the distance travelled
    def create_pickup_dropoff_dist_features(df):
        df['travel_distance'] = euc_distance(df['pickup_latitude'], 
                                             df['pickup_longitude'], 
                                             df['dropoff_latitude'],
                                             df['dropoff_longitude'])
        return df

    # create new column for the distance away from airports
    def create_airport_dist_features(df):
        airports = {'JFK_Airport': (-73.78,40.643),
                    'Laguardia_Airport': (-73.87, 40.77),
                    'Newark_Airport' : (-74.18, 40.69)}
        for k in airports:
            df['pickup_dist_'+k]=euc_distance(df['pickup_latitude'], 
                                              df['pickup_longitude'],
                                              airports[k][1],
                                              airports[k][0])
            df['dropoff_dist_'+k]=euc_distance(df['dropoff_latitude'], 
                                               df['dropoff_longitude'],
                                               airports[k][1],
                                               airports[k][0]) 
        return df

    df = create_time_features(df)
    df = create_pickup_dropoff_dist_features(df)
    df = create_airport_dist_features(df)
    df = df.drop(['key'], axis=1)
    return df

# %% [markdown]
# ##  Feature Scaling
# Making sure each feature has uniform scale

# %%
df_prescaled = df.copy()

# %%
df_scaled = df.drop(['fare_amount'], axis=1)


# %%
from sklearn.preprocessing import scale

df_scaled = scale(df_scaled)

# %%
cols = df.columns.tolist()
cols.remove('fare_amount')
df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)
df = df_scaled.copy()

# %% [markdown]
# # Splitting The Data: Keras

# %%
X = df.loc[:, df.columns != 'fare_amount'] 
y = df.loc[:, 'fare_amount']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation= 'relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1))

# %%
model.summary()

# %%
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(X_train, y_train, epochs=1)

# %%
def predict_random(df_prescaled, X_test, model):
    sample = X_test.sample(n=1, random_state=np.random.randint(low=0, 
                                                              high=10000))
    idx = sample.index[0]
  
    actual_fare = df_prescaled.loc[idx,'fare_amount']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
                 'Saturday', 'Sunday']
    day_of_week = day_names[df_prescaled.loc[idx,'day_of_week']]
    hour = df_prescaled.loc[idx,'hour']
    predicted_fare = model.predict(sample)[0][0]
    rmse = np.sqrt(np.square(predicted_fare-actual_fare))

    print("Trip Details: {}, {}:00hrs".format(day_of_week, hour))  
    print("Actual fare: ${:0.2f}".format(actual_fare))
    print("Predicted fare: ${:0.2f}".format(predicted_fare))
    print("RMSE: ${:0.2f}".format(rmse))

# %%
predict_random(df_prescaled, X_test, model)

# %%
predict_random(df_prescaled, X_test, model)

# %%


predict_random(df_prescaled, X_test, model)

# %%


predict_random(df_prescaled, X_test, model)

# %%
predict_random(df_prescaled, X_test, model)

# %%


predict_random(df_prescaled, X_test, model)

# %%
from sklearn.metrics import mean_squared_error

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Train RMSE: {:0.2f}".format(train_rmse))
print("Test RMSE: {:0.2f}".format(test_rmse))

# %%


# %% [markdown]
# # References
# We report here relevant references:
# Loy, J. (2019). Neural network projects with Python : the ultimate guide to using Python to explore the true power of neural networks through six projects. http://cds.cern.ch/record/2671438


