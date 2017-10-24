import csv
from datetime import datetime
import geopy.distance
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

################################################################################


feature_list=['order_id','R_Lon','R_Lat','C_Lon','C_Lat','order_pickedup_time','order_delivered_time']

def scaling_features(features_list):
    scaler = MinMaxScaler()
    scaled_name_feature_list = scaler.fit_transform(features_list)
    return scaled_name_feature_list


def targetFeatureSplit( data ):
    target = []
    features = []

    dist=[]
    time=[]
    time_order=[]
    for item in data:
        features.append([item['distance'],int(item['order_pickedup_time'].hour),item['R_Lat'],item['R_Lon'],item['C_Lat'],item['C_Lon'],item['order_pickedup_weekday']])

        dist.append(item['distance'])
        time_order.append(int(item['order_pickedup_time'].hour))
        time.append(item['time_taken_minutes'])



    time_np = np.array(time)
    time_np = time_np.reshape(-1,1)
    target = time_np
    return target, features

################################################################################

data_list=[]

with open("data_problem_1.csv", "r") as data_file:
    data_dict = csv.DictReader(data_file)

    for line in data_dict:
        data_list.append(line)

    for row in data_list:
        row['R_Lon'] = float(row['R_Lon'])
        row['R_Lat'] = float(row['R_Lat'])
        row['C_Lat'] = float(row['C_Lat'])
        row['C_Lon'] = float(row['C_Lon'])

        row['order_pickedup_time'] = datetime.strptime(row['order_pickedup_time'], "%Y-%m-%d %H:%M:%S")
        row['order_delivered_time'] = datetime.strptime(row['order_delivered_time'], "%Y-%m-%d %H:%M:%S")

        pickup_lat = row['R_Lat']
        pickup_lon = row['R_Lon']
        drop_lat = row['C_Lat']
        drop_lon = row['C_Lon']

        coords_1 = (pickup_lat, pickup_lon)
        coords_2 = (drop_lat, drop_lon)

        row['distance'] = geopy.distance.vincenty(coords_1, coords_2).km
        row['time_taken'] = row['order_delivered_time'] - row['order_pickedup_time']

        secs = row['time_taken'].total_seconds()
        hours = int(secs / 3600)
        minutes = int(secs / 60) % 60
        seconds = int(secs) % 60
        row['time_taken_hr'] = hours
        row['time_taken_minutes'] = minutes
        row['order_pickedup_time_hr'] = int(row['order_pickedup_time'].hour)
        row['order_pickedup_weekday'] = int(row['order_pickedup_time'].weekday())

        feature_list.append('distance')

        feature_list.append('order_pickedup_time_hr')

################################################################################

labels, features = targetFeatureSplit(data_list)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


cv = KFold(len(labels), 500, shuffle = True)

for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )


regression = LinearRegression()
regression.fit(features_train,labels_train)
predictions = regression.predict(features_test)

print 'R-square score: ',regression.score(features_test, labels_test)
print 'MEA(mean absolute error): ',mean_absolute_error(labels_test, predictions)

################################################################################

customer_lat = input('Please enter customer;s lat: ')
customer_lon = input('Please enter customer;s lon: ')

pickup_lat = input('Please enter pickup lat: ')
pickup_lon = input('Please enter pickup lon: ')

pickup_time = raw_input('Please enter pickup_time: ')

pickup = datetime.strptime(str(pickup_time), "%Y-%m-%d %H:%M:%S")
pickup_hr = pickup.hour
pickup_day = int(pickup.weekday())

coords_1 = (pickup_lat, pickup_lon)
coords_2 = (customer_lat, customer_lon)
dist = geopy.distance.vincenty(coords_1, coords_2).km

features = [(float(dist),pickup_hr,float(customer_lat),float(customer_lon),float(pickup_lat),float(pickup_lon),pickup_day)]

predict = regression.predict(features)
print 'Time taken: ',predict
