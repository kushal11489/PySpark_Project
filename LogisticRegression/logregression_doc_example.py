from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

sample_test_data_path = 'test_input/logistic_regression/sample_libsvm_data.txt'
spark = SparkSession.builder.appName('mylogreg').getOrCreate()

data = spark.read.format('libsvm').load(sample_test_data_path)

train_data, test_data = data.randomSplit([0.7,0.3])

mylogreg_model = LogisticRegression()

fitted_log_reg_model = mylogreg_model.fit(train_data)

# log_summary = fitted_log_reg_model.summary
#
# log_summary.predictions.show()

prediction_and_labels = fitted_log_reg_model.evaluate(test_data)

prediction_and_labels.predictions.show()

my_eval = BinaryClassificationEvaluator()

my_final_roc = my_eval.evaluate(prediction_and_labels.predictions)

print my_final_roc
