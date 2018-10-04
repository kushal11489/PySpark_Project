from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

sample_test_data_path = 'test_input/logistic_regression/titanic.csv'
spark = SparkSession.builder.appName('titanic').getOrCreate()

data = spark.read.csv(sample_test_data_path, inferSchema=True, header=True)

my_cols = data.select(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

my_final_data = my_cols.na.drop()

gender_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')

# A B C
# 0 1 2
# ONE HOT ENCODE
# EXAMPLE A
# [1, 0, 0]

gender_encoder = OneHotEncoder(inputCol='SexIndex', outputCol='SexVec')

embark_indexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex', outputCol='EmbarkVec')

assembler = VectorAssembler(inputCols=['Pclass', 'SexVec', 'EmbarkVec', 'Age', 'SibSp', 'Parch', 'Fare'],
                            outputCol='features')

log_reg_titanic = LogisticRegression(featuresCol='features', labelCol='Survived')

pipeline = Pipeline(stages=[gender_indexer, embark_indexer,
                            gender_encoder, embark_encoder,
                            assembler, log_reg_titanic])

train_data, test_data = my_final_data.randomSplit([0.7,0.3])

fit_model = pipeline.fit(train_data)

results = fit_model.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Survived')

results.select('Survived', 'prediction').show()

# Area under curve(ROC)
AUC = my_eval.evaluate(results)

print AUC
