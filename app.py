from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, when, abs
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import DecisionTreeRegressionModel, GBTRegressionModel, RandomForestRegressionModel, LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import os
import time

def validate_input_file(spark, path, required_columns):
    """
    Validate the input file for common errors like empty file, missing columns, or incorrect format.
    """
    try:
        # Attempt to read the file
        df = spark.read.csv(path, header=True, inferSchema=True)

        # Check if the file is empty
        if df.count() == 0:
            print(f"Error: The input file at {path} is empty.")
            sys.exit(1)

        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns in the input file: {missing_columns}")
            sys.exit(1)

        return df
    except Exception as e:
        print(f"Error: Unable to read the input file at {path}. Ensure it is a valid CSV file.\nDetails: {e}")
        sys.exit(1)


def preprocess_test_data(test_data,spark):
    """
    Applies the same preprocessing as models.py to the test data
    """

    # Load the index files (for the categorical variables and to match what the model recognizes).
    carrierMapping=spark.read.format("csv").option("inferschema", "true").option("header", "true").load("mappings/UniqueCarrier_mapping.csv")
    carrierMapping=carrierMapping.withColumn("UniqueCarrier",col("label"))
    carrierMapping=carrierMapping.withColumn("UniqueCarrierIndex",col("index"))

    
    planeModelMapping=spark.read.format("csv").option("inferschema", "true").option("header", "true").load("mappings/PlaneModel_mapping.csv")
    planeModelMapping=planeModelMapping.withColumn("PlaneModel",col("label"))
    planeModelMapping=planeModelMapping.withColumn("PlaneModelIndex",col("index"))


    originMapping=spark.read.format("csv").option("inferschema", "true").option("header", "true").load("mappings/Origin_mapping.csv")
    originMapping=originMapping.withColumn("Origin",col("label"))
    originMapping=originMapping.withColumn("OriginIndex",col("index"))


    # Validate and filter null values
    test_data = test_data.filter((col("DepTime").isNotNull()) & (col("CRSDepTime").isNotNull()))

    # Cargar el archivo plane-data.csv
    airplane_info = spark.read.format("csv").option("inferschema", "true").option("header", "true").load("data/plane-data.csv")
    airplane_info = airplane_info.dropna()

    # Perform the join with plane-data.csv
    test_data = test_data.join(airplane_info, on=['TailNum'], how='inner')

    # Perform the join with the mappings
    test_data=test_data.join(carrierMapping,on=['UniqueCarrier'],how="inner")
    test_data=test_data.join(planeModelMapping,on=['PlaneModel'],how="inner")
    test_data=test_data.join(originMapping,on=['Origin'],how="inner")

    # Cast certain columns to proper data types
    test_data = test_data.withColumn('CRSElapsedTime', col('CRSElapsedTime').cast(DoubleType()))
    test_data = test_data.withColumn('DepDelay', col('DepDelay').cast(DoubleType()))
    test_data = test_data.withColumn('TaxiOut', col('TaxiOut').cast(DoubleType()))
    test_data = test_data.withColumn('Distance', col('Distance').cast(DoubleType()))
    test_data = test_data.withColumn('ArrDelay', col('ArrDelay').cast(DoubleType()))
    test_data = test_data.withColumn('DepTime', col('DepTime').cast(DoubleType()))
    test_data = test_data.withColumn('CRSDepTime', col('CRSDepTime').cast(DoubleType()))
    test_data = test_data.withColumn('CRSArrTime', col('CRSArrTime').cast(IntegerType()))
    test_data = test_data.withColumn('PlaneYear', col('PlaneYear').cast(IntegerType()))

    test_data = test_data.withColumn("PlaneAgeYears", col("Year") - col("PlaneYear"))

    test_data=test_data.withColumn("DepTimeMinutes",col("DepTime")%100)
    test_data=test_data.withColumn("DepTimeHour",col("DepTime")/100)

    test_data=test_data.withColumn("DepTimeHour",test_data["DepTimeHour"].cast("int"))

    test_data=test_data.withColumn("DepTimeMinutesSinceMidnight",test_data["DepTimeHour"]+test_data["DepTimeMinutes"])

    test_data=test_data.withColumn("CRSDepTimeMinutes",col("CRSDepTime")%100)
    test_data=test_data.withColumn("CRSDepTimeHour",col("CRSDepTime")/100)

    test_data=test_data.withColumn("CRSDepTimeMinutesSinceMidnight",test_data['CRSDepTimeMinutes']+test_data['CRSDepTimeHour'])

    test_data=test_data.filter((col('ArrDelay') > -120) & (col('ArrDelay') < 300))

    # Scale numerical columns
    for col_name in ["Distance", "DepTimeMinutesSinceMidnight", "CRSDepTimeMinutesSinceMidnight"]:
        assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}Vec")
        test_data = assembler.transform(test_data)
        scaler = StandardScaler(inputCol=f"{col_name}Vec", outputCol=f"{col_name}Scaled", withStd=True, withMean=False)
        scaler_model = scaler.fit(test_data)
        test_data = scaler_model.transform(test_data)

    # Assemble final feature vector
    feature_columns = [
        'Month', 'DayofMonth', 'DayOfWeek', 'OriginIndex',"PlaneAgeYears","PlaneModelIndex",
        'DepTimeMinutesSinceMidnightScaled', 'CRSDepTimeMinutesSinceMidnightScaled',
        'CRSArrTime', 'UniqueCarrierIndex', 'CRSElapsedTime',
        'DepDelay', 'DistanceScaled', 'TaxiOut'
    ]

    cols_to_keep=['Month', 'DayofMonth', 'DayOfWeek', 'OriginIndex',"PlaneAgeYears","PlaneModelIndex",
        'DepTimeMinutesSinceMidnightScaled', 'CRSDepTimeMinutesSinceMidnightScaled',
        'CRSArrTime', 'UniqueCarrierIndex', 'CRSElapsedTime',
        'DepDelay', 'DistanceScaled', 'TaxiOut','ArrDelay']

    test_data=test_data.select(cols_to_keep)

    test_data=test_data.dropna()

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    test_data = assembler.transform(test_data)

    return test_data

def load_best_model():
    """
    Load the best model from the directory and determine its type.
    """
    model_mapping = {
        "DecisionTreeRegressionModel": DecisionTreeRegressionModel,
        "GBTRegressionModel": GBTRegressionModel,
        "RandomForestRegressionModel": RandomForestRegressionModel,
        "LinearRegressionModel": LinearRegressionModel
    }

    model_folders = {
        "DecisionTreeRegressionModel": "DecisionTreeRegresspr",
        "GBTRegressionModel": "GBTRegressor",
        "RandomForestRegressionModel": "RandomForestRegressor",
        "LinearRegressionModel": "LinearRegression"
    }

    # Path to models directory
    models_dir = "best_model"

    for model_type, model_class in model_mapping.items():
        model_folder=model_folders[model_type]   
        model_path = os.path.join(models_dir,model_folder)
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            return model_class.load(model_path), model_type

    print("No valid model found in the directory.")
    sys.exit(1)


def evaluate_model(model, test_data, model_name, metrics_file, predictions_file,file_path):
    """
    Evaluate a model and save the predictions and metrics.
    """
    print(f"Evaluating model: {model_name}")
    predictions = model.transform(test_data)

    # Adjust negative predictions to zero
    predictions = predictions.withColumn(
        "prediction",
        when(predictions["prediction"] < 0, 0).otherwise(predictions["prediction"])
    )

    # Evaluators
    evaluator_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")
    evaluator_mae = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    # Handle negative predictions and avoid division by zero in MAPE
    predictions = predictions.withColumn("prediction", col("prediction").cast(DoubleType()))
    predictions = predictions.withColumn("absolute_error", abs(col("prediction") - col("ArrDelay")))
    predictions = predictions.withColumn("ArrDelay", col("ArrDelay").cast(DoubleType()))
    mape = predictions.filter(col("ArrDelay") != 0).selectExpr("avg(absolute_error / ArrDelay) as mape").collect()[0]["mape"] * 100

    # Save metrics to a file
    os.makedirs("results", exist_ok=True)
    model_results_dir = f"results/{model_name}"
    os.makedirs(model_results_dir, exist_ok=True)

    # Save metrics to a file inside the model's folder
    metrics_path = os.path.join(model_results_dir, "metrics.txt")

    try:
        output_string="Dataset: "+str(file_path)+"\n"
        output_string+="Model: "+str(model_name)+"\n"
        output_string+="Root Mean Squared Error (RMSE): "+str(rmse)+"\n"
        output_string+="R-Squared (R2): "+str(r2)+"\n"
        output_string+="Mean Absolute Error (MAE): "+str(mae)+"\n"
        output_string+="Mean Absolute Percentage Error (MAPE): "+str(mape)+"\n"
        with open(metrics_file, "w") as f:
            f.write(output_string)
            f.close()
        print(f"Metrics appended to {metrics_file}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    # Save predictions to a folder inside the model's directory
    try:
        predictions.select("ArrDelay", "prediction").write.csv(predictions_file, header=True, mode="overwrite")
        print(f"Predictions saved successfully at {predictions_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")



def main():
    start_time = time.time()

    # Create a Spark session
    spark = SparkSession.builder.appName("Spark Model Testing").getOrCreate()

    # Check for input test data path
    if len(sys.argv) < 2:
        print("Usage: spark-submit app.py <path_to_test_data>")
        sys.exit(1)

    test_data_path = sys.argv[1:]

    print(f"Validating and loading test data from: {test_data_path}")
    required_columns = ["ArrDelay"]  # Add required columns
    

    # Setup results directory and files
    results_dir = "results2"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the best model
    best_model, model_name = load_best_model()

    for file_path in test_data_path:
        file_name = os.path.basename(file_path).split(".")[0]
        metrics_file = os.path.join(results_dir, f"metrics_{file_name}.txt")
        predictions_file = os.path.join(results_dir, f"predictions_{file_name}")

        # Clear existing results
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
        if os.path.exists(predictions_file):
            os.system(f"rm -rf {predictions_file}")

        test_data = validate_input_file(spark, file_path, required_columns)

        # Preprocess test data
        test_data = preprocess_test_data(test_data,spark)

        # Evaluate the model
        evaluate_model(best_model, test_data, model_name,metrics_file, predictions_file,file_path)

    # Show execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main()