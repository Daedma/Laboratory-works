#
# Лабораторная 1. Неинтерактивный анализ данных велопарковок SF Bay Area Bike Share в Apache Spark
#
# Для каждой стартовой станции найдем среднее время поездки и самую первую поездку.
# 
# Команда для запуска
# spark-submit --master yarn --deploy-mode cluster L1_noninteractive_bike_analysis.py 
# 
# используйте --conf spark.yarn.submit.waitAppCompletion=true, чтобы ждать завершения выполнения спарк задачи
# 

from pyspark import SparkContext, SparkConf
from typing import NamedTuple
from datetime import datetime
import numpy as np

conf = SparkConf().setAppName("Lab1_Script")
# conf.set("spark.yarn.submit.waitAppCompletion", "false")

sc = SparkContext(conf=conf)

tripData = sc.textFile("trips.csv")
# запомним заголовок, чтобы затем его исключить из данных
tripsHeader = tripData.first()
trips = tripData.filter(lambda row: row != tripsHeader).map(lambda row: row.split(",", -1))

stationData = sc.textFile("stations.csv")
stationsHeader = stationData.first()
stations = stationData.filter(lambda row: row != stationsHeader).map(lambda row: row.split(",", -1))

def initStation(stations):
    class Station(NamedTuple):
        station_id: int
        name: str
        lat: float
        long: float
        dockcount: int
        landmark: str
        installation: str
    
    for station in stations:
        yield Station(
            station_id = int(station[0]),
            name = station[1],
            lat = float(station[2]),
            long = float(station[3]),
            dockcount = int(station[4]),
            landmark = station[5],
            installation = datetime.strptime(station[6], '%m/%d/%Y')
        )
        
def initTrip(trips):
    class Trip(NamedTuple):
        trip_id: int
        duration: int
        start_date: datetime
        start_station_name: str
        start_station_id: int
        end_date: datetime
        end_station_name: str
        end_station_id: int
        bike_id: int
        subscription_type: str
        zip_code: str
        
    for trip in trips:
        yield Trip(                             
             trip_id = int(trip[0]),
             duration = int(trip[1]),
             start_date = datetime.strptime(trip[2], '%m/%d/%Y %H:%M'),
             start_station_name = trip[3],
             start_station_id = int(trip[4]),
             end_date = datetime.strptime(trip[5], '%m/%d/%Y %H:%M'),
             end_station_name = trip[6],
             end_station_id = trip[7],
             bike_id = int(trip[8]),
             subscription_type = trip[9],
             zip_code = trip[10]
        )
        
stationsInternal = stations.mapPartitions(initStation)
tripsInternal = trips.mapPartitions(initTrip)

tripsByStartStation = tripsInternal.keyBy(lambda trip: trip.start_station_name)

# среднее время поездок стартовой станции

def seqFunc(acc, duration):
    duration_sum, count = acc
    return (duration_sum + duration, count + 1)

def combFunc(acc1, acc2):
    duration_sum1, count1 = acc1
    duration_sum2, count2 = acc2
    return (duration_sum1+duration_sum2, count1+count2)

def meanFunc(acc):
    duration_sum, count = acc
    return duration_sum/count

avgDurationByStartStation = tripsByStartStation\
  .mapValues(lambda trip: trip.duration)\
  .aggregateByKey(
    zeroValue=(0,0),
    seqFunc=seqFunc,
    combFunc=combFunc)\
  .mapValues(meanFunc)\
  .sortBy(lambda x: x[1], ascending=False)

#
# первые поездки стартовых станций 
#

firstStationsTrip = tripsByStartStation\
  .reduceByKey(lambda tripA, tripB: tripA if tripA.start_date < tripB.start_date else tripB)

#
# старт выполнения задач и сохранение результата в HDFS
#

avgDurationByStartStation.saveAsTextFile("avg_duration_of_start_stations")
firstStationsTrip.saveAsTextFile("first_station_trips")

# результаты сохранятся в директории распределённой файловой системы в домашней папке пользователя /user/$USER/:
#  - avg_duration_of_start_stations
#  - first_station_trips
# 
# получить результаты в виде одного файла можно командой `hadoop fs -getmerge`:
#   hadoop fs -getmerge avg_duration_of_start_stations avg_duration_of_start_stations.txt
#   hadoop fs -getmerge first_station_trips first_station_trips.txt

sc.stop()