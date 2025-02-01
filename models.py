
from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor,GBTRegressor,RandomForestRegressor,LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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

dataFrame = spark.read.parquet("data_cleaned.parquet")

# Split the data into training and test sets (20% held out for testing)
(trainingData, testData) = dataFrame.randomSplit([0.8, 0.2],seed=42)


## Iterate through different models 
model_options={"LinearRegression":LinearRegression(featuresCol="features",labelCol="ArrDelay"),
                "DecisionTreeRegressor":DecisionTreeRegressor(featuresCol="features",labelCol="ArrDelay"),
                "GBTRegressor":GBTRegressor(featuresCol="features",labelCol="ArrDelay"),
                "RandomForestRegressor":RandomForestRegressor(featuresCol="features",labelCol="ArrDelay")
                
}

best_model = None
best_model_name = None
best_rmse = float("inf")

for a_model in model_options.keys():

    # Train Model
    model = model_options[a_model]

    
    if a_model=="LinearRegression":
        paramGrid = ParamGridBuilder() \
        .addGrid(model.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    elif a_model=="GBTRegressor":
        paramGrid = ParamGridBuilder() \
        .addGrid(model.maxDepth, [3, 5, 7]) \
        .addGrid(model.maxBins, [300, 400]) .build()
        # .addGrid(model.stepSize, [0.01, 0.1, 0.2]) \
        # .addGrid(model.maxIter, [50, 100]) \

    elif a_model=="RandomForestRegressor":
        paramGrid = ParamGridBuilder() \
        .addGrid(model.maxDepth, [3, 5, 10]) \
        .addGrid(model.numTrees, [50, 100, 200]) \
        .addGrid(model.maxBins, [300,400]) \
        .build()
        # .addGrid(model.subsamplingRate, [0.7, 1.0]) \
        # .addGrid(model.featureSubsetStrategy, ["auto", "sqrt", "log2"]) \
    
    elif a_model=="DecisionTreeRegressor":
        paramGrid = ParamGridBuilder() \
        .addGrid(model.maxDepth, [3, 5, 10]) \
        .addGrid(model.maxBins, [300,400]) \
        .build()
        # .addGrid(model.minInstancesPerNode, [10]) \
        # .addGrid(model.minInfoGain, [0.0, 0.01, 0.05]) \

    # Evaluate Model
    evaluator=RegressionEvaluator(metricName="rmse",labelCol="ArrDelay",predictionCol="prediction")

    crossval = CrossValidator(estimator=model,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
           numFolds=5)  
    
    cvModel = crossval.fit(trainingData)

    # Predicting using adjusted model
    bestModel = cvModel.bestModel
    predictions = bestModel.transform(testData)

    print(f"BestModel type: {type(bestModel)}")
    test_predictions = bestModel.transform(testData)
    rmse = evaluator.evaluate(test_predictions)

    # Evaluate model
    rmse = evaluator.evaluate(predictions)

    # Compare best model with current model
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = bestModel
        best_model_name = a_model

# Store best model
if best_model is not None:
    # Store best model adjusted
    bestModel.write().overwrite().save(f"best_model/{a_model}")

    with open('results.txt','a') as f:
        f.write(a_model+f" Root Mean Squared Error (RMSE) on test data: {best_rmse}+\n")

spark.stop()