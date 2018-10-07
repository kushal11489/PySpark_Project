from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName('college').getOrCreate()

sample_test_data_path = 'test_input/trees/College.csv'

data = spark.read.csv(sample_test_data_path, inferSchema=True, header=True)

assembler = VectorAssembler(inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad'
                                       , 'Outstate', 'Room_Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni',
                                       'Expend', 'Grad_Rate'], outputCol='features')

output = assembler.transform(data)

indexer = StringIndexer(inputCol='Private', outputCol='PrivateIndex')

output_fixed = indexer.fit(output).transform(output)

final_data = output_fixed.select('features', 'PrivateIndex')

train_data, test_data = final_data.randomSplit([0.7, 0.3])

dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol='features')
rfc = RandomForestClassifier(labelCol='PrivateIndex', featuresCol='features', numTrees=100)
gbtc = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')

dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbtc_model = gbtc.fit(train_data)

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbtc_preds = gbtc_model.transform(test_data)

my_binary_eval = BinaryClassificationEvaluator(labelCol='PrivateIndex')

print 'DTC:'
print my_binary_eval.evaluate(dtc_preds)

print 'RFC:'
print my_binary_eval.evaluate(rfc_preds)

my_binary_eval2 = BinaryClassificationEvaluator(labelCol='PrivateIndex', rawPredictionCol='prediction')
print 'GBTC:'
print my_binary_eval.evaluate(gbtc_preds)

