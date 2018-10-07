from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
# We can use trees method for regression too!
from pyspark.ml.regression import RandomForestRegressor

spark = SparkSession.builder.appName('mytree').getOrCreate()

sample_test_data_path = 'test_input/trees/sample_libsvm_data.txt'

data = spark.read.format('libsvm').load(sample_test_data_path)

train_data, test_data = data.randomSplit([0.7, 0.3])

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier(numTrees=100)
gbtc = GBTClassifier()

dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbtc_model = gbtc.fit(train_data)

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbtc_preds = gbtc_model.transform(test_data)

acc_eval = MulticlassClassificationEvaluator(metricName='accuracy')

print 'DTC Accuracy:'
acc_eval.evaluate(dtc_preds)

print 'RFC Accuracy:'
acc_eval.evaluate(rfc_preds)

print 'GBTC Accuracy:'
acc_eval.evaluate(gbtc_preds)
