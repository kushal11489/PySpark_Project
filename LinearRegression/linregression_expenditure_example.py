from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler


sample_test_data_path = 'test_input/linear_regression/Ecommerce_Customers.csv'
spark = SparkSession.builder.appName('lr_example').getOrCreate()

data = spark.read.csv(sample_test_data_path, inferSchema=True, header=True)

assembler = VectorAssembler(inputCols=['Avg Session Length',
                                       'Time on App',
                                       'Time on Website',
                                       'Length of Membership'], outputCol='features')

output = assembler.transform(data)

final_data = output.select('features', 'Yearly Amount Spent')

train_data, test_data = final_data.randomSplit([0.7, 0.3])

lr = LinearRegression(featuresCol='features', labelCol='Yearly Amount Spent', predictionCol='prediction')

lr_model = lr.fit(train_data)

test_results = lr_model.evaluate(test_data)

# residuals value

test_results.residuals.show()
print test_results.rootMeanSquaredError
print test_results.r2

unlabelled_data = test_data.select('features')

predictions = lr_model.transform(unlabelled_data)
predictions.show()
