### SF Bay Area Bike Share

https://www.kaggle.com/benhamner/sf-bay-area-bike-share

stations.csv  
схема:
```
id: station ID number
name: name of station
lat: latitude
long: longitude
dock_count: number of total docks at station
city: city (San Francisco, Redwood City, Palo Alto, Mountain View, San Jose)
installation_date: original date that station was installed. If station was moved, it is noted below.
```

trips.csv  
схема:
```
id: numeric ID of bike trip
duration: time of trip in seconds
start_date: start date of trip with date and time, in PST
start_station_name: station name of start station
start_station_id: numeric reference for start station
end_date: end date of trip with date and time, in PST
end_station_name: station name for end station
end_station_id: numeric reference for end station
bike_id: ID of bike used
subscription_type: Subscriber = annual or 30-day member; Customer = 24-hour or 3-day member
zip_code: Home zip code of subscriber (customers can choose to manually enter zip at kiosk however data is unreliable)
```
### Stack Overflow Data Dump

https://archive.org/details/stackexchange

posts_sample.xml

```
sc.textFile("posts.xml").mapPartitions(_.take(1000)).repartition(1).saveAsTextFile("posts_sample.xml")
```

### New York City Taxi Data(2010-2013)

https://databank.illinois.edu/datasets/IDB-9610843 или https://uofi.app.box.com/v/NYCtaxidata 

nyctaxi.csv   
схема: https://uofi.app.box.com/v/NYCtaxidata/file/33670345557
```
_id: primary key
_rev:  unknown attribute.
medallion: a permit to operate a yellow taxicab in New York City, it is effectively a (randomly assigned) car ID. See also medallions.
hack_license: a license to drive the vehicle, it is effectively a (randomly assigned) driver ID. See also hack license.
vendor_id: e.g.,Verifone Transportation Systems(VTS), or Mobile KnowledgeSystems Inc(CMT), implemented as part of theTechnology Passenger Enhancements Project.
rate_code: taximeter rate, see NYCT&L description.
store_and_fwd_flag: unknown attribute.
pickup_datetime: start time of the trip, mm-dd-yyyy hh24:mm:ss EDT.
dropoff_datetime: end time of the trip, mm-dd-yyyy hh24:mm:ss EDT.
passenger_count: number of passengers on the trip, default value is one.
trip_time in secs: trip time measured by the taximeter in seconds.
trip_distance: trip distance measured by the taximeter in miles.
pickup_longitude and pickup_latitude: GPS coordinates at the start of the trip.
dropoff_longitude and dropoff_latitude: GPS coordinates at the end of the trip.
```


nycTaxiFares.gz   
nycTaxiRides.gz  
схема: https://github.com/apache/flink-training/blob/master/README.md#schema-of-taxi-ride-events

### List of programming languages 

https://en.wikipedia.org/wiki/List_of_programming_languages 

programming-languages.csv


### Другие источники данных

https://github.com/infoculture/awesome-opendata-rus  
