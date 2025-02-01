
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
import pandas as pd

spark = SparkSession \
    .builder \
    .appName("Machine Learning") \
    .config('spark.executor.memory', '8g') \
    .config('spark.driver.memory', '8g') \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config('spark.executor.cores', '4')  \
    .config('spark.executor.instances', '4') \
    .config('spark.sql.shuffle.partitions', '400')  \
    .config('spark.memory.fraction', '0.8') \
    .config('spark.memory.storageFraction', '0.5') \
    .config('spark.dynamicAllocation.enabled', 'true')  \
    .config('spark.dynamicAllocation.minExecutors', '1') \
    .config('spark.dynamicAllocation.maxExecutors', '10') \
    .config('spark.dynamicAllocation.initialExecutors', '2') \
    .config('spark.executor.memoryOverhead', '4g') \
    .getOrCreate()


dataFrame = spark.read.format("csv").option("inferschema","true").option("header", "true").load("data/2007.csv")

## Load Airplane age and model type as well as first service date
airplaneInfo= spark.read.format("csv").option("inferschema","true").option("header", "true").load("data/plane-data.csv")

airplaneInfo=airplaneInfo.dropna()

dataFrame=dataFrame.join(airplaneInfo,on=['TailNum'],how='inner')


## Cast certain columns to proper integer datatype
dataFrame=dataFrame.withColumn('CRSElapsedTime',dataFrame['CRSElapsedTime'].cast("double"))
dataFrame=dataFrame.withColumn('DepDelay',dataFrame['DepDelay'].cast("double"))
dataFrame=dataFrame.withColumn('TaxiOut',dataFrame['TaxiOut'].cast("double"))
dataFrame=dataFrame.withColumn('Distance',dataFrame['Distance'].cast("double"))
dataFrame=dataFrame.withColumn('ArrDelay',dataFrame.ArrDelay.cast("double"))
dataFrame=dataFrame.withColumn('DepTime',dataFrame['DepTime'].cast("integer"))
dataFrame=dataFrame.withColumn('CRSDepTime',dataFrame['CRSDepTime'].cast("integer"))
dataFrame=dataFrame.withColumn('CRSArrTime',dataFrame['CRSArrTime'].cast("integer"))
dataFrame=dataFrame.withColumn('PlaneYear',dataFrame['PlaneYear'].cast("integer"))
dataFrame=dataFrame.withColumn('ArrDelay',dataFrame['ArrDelay'].cast("double"))


## Infer Plane age in years 
dataFrame=dataFrame.withColumn("PlaneAgeYears",col("Year")-col("PlaneYear"))

## Replace any null values with 
dataFrame = dataFrame.fillna(0)

## Apply to DepTime
dataFrame=dataFrame.withColumn("DepTimeMinutes",col("DepTime")%100)
dataFrame=dataFrame.withColumn("DepTimeHour",60*col("DepTime")/100)

## Cast to integer
dataFrame=dataFrame.withColumn("DepTimeHour",dataFrame["DepTimeHour"].cast("int"))

## Add columns together to get total minutes
dataFrame=dataFrame.withColumn("DepTimeMinutesSinceMidnight",dataFrame["DepTimeHour"]+dataFrame["DepTimeMinutes"])

## Apply to CRSDepTime
dataFrame=dataFrame.withColumn("CRSDepTimeMinutes",col("CRSDepTime")%100)
dataFrame=dataFrame.withColumn("CRSDepTimeHour",60*col("CRSDepTime")/100)

## Cast to integer 
dataFrame=dataFrame.withColumn("CRSDepTimeHour",dataFrame["CRSDepTimeHour"].cast("int"))

## Add columns together to get total  minutes
dataFrame=dataFrame.withColumn("CRSDepTimeMinutesSinceMidnight",dataFrame['CRSDepTimeMinutes']+dataFrame['CRSDepTimeHour'])

## Apply a filter to ArrDelay 
dataFrame=dataFrame.filter((col('ArrDelay') > -120) & (col('ArrDelay') < 300))


## Filtering out forbidden columns and get rid of exceptional flights (cancelled or diverted)

## For now, dataFrameYear will also be excluded dataFrame['Year'], dataFrame['FlightNum'], dataFrame['TailNum']

dataFrame=dataFrame.select(dataFrame['Month'],dataFrame['DayOfMonth'],dataFrame['DayOfWeek'], dataFrame['Origin'],
                           dataFrame['DepTimeMinutesSinceMidnight'], dataFrame['CRSDepTimeMinutesSinceMidnight'],
                           dataFrame['CRSDepTimeHour'],dataFrame['CRSArrTime'],dataFrame['PlaneModel'],dataFrame['PlaneEngineType'],dataFrame['PlaneAgeYears'],
                           dataFrame['UniqueCarrier'],dataFrame['CRSElapsedTime'], dataFrame['DepDelay'],
                           dataFrame['Distance'],dataFrame['TaxiOut'],dataFrame['ArrDelay']).filter((dataFrame['Cancelled']!=1) & (dataFrame['Diverted']!=1))

## Start ML Pre-Processing
## Scale Distance value
assembler = VectorAssembler(inputCols=["Distance"], outputCol="DistanceVec")
dataFrame = assembler.transform(dataFrame)

from pyspark.ml.feature import StandardScaler

## Define scaling
scaler = StandardScaler(inputCol="DistanceVec", outputCol="DistanceScaled",
                        withStd=True, withMean=False)

## Fit scaler to dataset
scaler_model = scaler.fit(dataFrame)

## Apply scaler
dataFrame = scaler_model.transform(dataFrame)

## Scale Time Variables (DepTime)
assembler = VectorAssembler(inputCols=["DepTimeMinutesSinceMidnight"], outputCol="DepTimeMinutesSinceMidnightVec")
dataFrame = assembler.transform(dataFrame)

scaler = StandardScaler(inputCol="DepTimeMinutesSinceMidnightVec", outputCol="DepTimeMinutesSinceMidnightScaled",
                        withStd=True, withMean=False)

scaler_model = scaler.fit(dataFrame)

dataFrame = scaler_model.transform(dataFrame)


## Scale Time Variables (CRSDepTime)
assembler = VectorAssembler(inputCols=["CRSDepTimeMinutesSinceMidnight"], outputCol="CRSDepTimeMinutesSinceMidnightVec")
dataFrame = assembler.transform(dataFrame)

scaler = StandardScaler(inputCol="CRSDepTimeMinutesSinceMidnightVec", outputCol="CRSDepTimeMinutesSinceMidnightScaled",
                        withStd=True, withMean=False)

scaler_model = scaler.fit(dataFrame)

dataFrame = scaler_model.transform(dataFrame)

## Scale Time Variables (CRSElapsedTime)
assembler = VectorAssembler(inputCols=["CRSElapsedTime"], outputCol="CRSElapsedTimeVec")
dataFrame = assembler.transform(dataFrame)


scaler = StandardScaler(inputCol="CRSElapsedTimeVec", outputCol="CRSElapsedTimeScaled",
                        withStd=True, withMean=False)

scaler_model = scaler.fit(dataFrame)

dataFrame = scaler_model.transform(dataFrame)


# Apply indexing to categorical columns Origin, UniqueCarrier, PlaneModel
for categorical_variable in ['Origin','UniqueCarrier','PlaneModel']:
    indexer=StringIndexer(inputCol=categorical_variable, outputCol=categorical_variable+"Index")
    indexer_model = indexer.fit(dataFrame)
    
    dataFrame=indexer_model.transform(dataFrame)

    # Save the mapping
    labels = indexer_model.labels  # List of original strings
    
    labels_df=pd.DataFrame(labels)

    labels_df.columns=['label']

    labels_df['index']=labels_df.index

    labels_df.to_csv('mappings/'+categorical_variable+'_mapping.csv',index=False)

## Reduce junk columns 
selected_cols=['Month', 'DayOfMonth', 'DayOfWeek', 'OriginIndex', 'DepTimeMinutesSinceMidnightScaled', "PlaneAgeYears","PlaneModelIndex",
                             'CRSDepTimeMinutesSinceMidnightScaled','CRSArrTime', 'UniqueCarrierIndex', 'CRSElapsedTimeScaled', 
                             'DepDelay', 'DistanceScaled', 'TaxiOut','ArrDelay']

dataFrame=dataFrame.select(selected_cols)


vecAssembler=VectorAssembler(inputCols=['Month', 'DayOfMonth', 'DayOfWeek', 'OriginIndex', 'DepTimeMinutesSinceMidnightScaled',"PlaneAgeYears","PlaneModelIndex",
                             'CRSDepTimeMinutesSinceMidnightScaled','CRSArrTime', 'UniqueCarrierIndex', 'CRSElapsedTimeScaled', 
                             'DepDelay', 'DistanceScaled', 'TaxiOut'],outputCol="features")


output = vecAssembler.transform(dataFrame)

output.write.mode("overwrite").parquet("data_cleaned.parquet")
print("success")

spark.stop()