from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler, StringIndexer

sample_test_data_path = 'test_input/linear_regression/cruise_ship_info.csv'
spark = SparkSession.builder.appName('hyundai_project').getOrCreate()

data = spark.read.csv(sample_test_data_path, inferSchema=True, header=True)

indexer = StringIndexer(inputCol='Cruise_line', outputCol='cruise_cat')

new_data = indexer.fit(data).transform(data)

assembler = VectorAssembler(inputCols=['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density',
                                       'cruise_cat'], outputCol='features')

output = assembler.transform(new_data)

final_data = output.select('features', 'crew')

train_data, test_data = final_data.randomSplit([0.7,0.3])

lr = LinearRegression(labelCol='crew')

lr_model = lr.fit(train_data)

test_results = lr_model.evaluate(test_data)

# residuals value

test_results.residuals.show()
print test_results.rootMeanSquaredError
print test_results.r2

unlabelled_data = test_data.select('features')

predictions = lr_model.transform(unlabelled_data)
predictions.show()
