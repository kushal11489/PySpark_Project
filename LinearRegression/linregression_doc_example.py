from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
sample_test_data_path = 'test_input/linear_regression/sample_linear_regression_data.txt'
spark = SparkSession.builder.appName('lrex').getOrCreate()

all_data = spark.read.format('libsvm').load(sample_test_data_path)

# Split the data into training and test
training_data, test_data = all_data.randomSplit([0.7, 0.3])

# Initialize model
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')

# Fit the model
lrModel = lr.fit(training_data)

test_results = lrModel.evaluate(test_data)

rms = test_results.rootMeanSquaredError
print rms
# Unlabelled data

unlabelled_data = test_data.select('features')

predictions = lrModel.transform(unlabelled_data)

print predictions
