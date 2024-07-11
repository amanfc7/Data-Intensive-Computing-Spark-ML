from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer, StringIndexer, ChiSqSelector
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder

# initializing Spark context:

sc = SparkContext(appName="TextClassification")
spark = SparkSession.builder.appName("TextClassification").getOrCreate()

# Loading the stopwords:

stopwords = set()
with open("src/stopwords.txt", "r") as f:
    for line in f:
        stopwords.add(line.strip())

# Loading and preprocessing the dataset:

def load_and_preprocess_data(file_path):
    data = spark.read.json(file_path)
    data = data.withColumn("text", lower(col("reviewText")))
    data = data.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))
    return data.select(col("category"), col("text"))

# to load data from reviews dataset:

data = load_and_preprocess_data("hdfs:///user/dic24_shared/amazon-reviews/full/reviews_devset.json")

# to define stages of the pipeline:

indexer = StringIndexer(inputCol="category", outputCol="label")
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=list(stopwords))
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=2000)
idf = IDF(inputCol="raw_features", outputCol="features")
selector = ChiSqSelector(numTopFeatures=2000, featuresCol="features", outputCol="selected_features", labelCol="label")
normalizer = Normalizer(inputCol="selected_features", outputCol="normalized_features", p=2.0)
svm = LinearSVC(featuresCol="normalized_features", labelCol="label")
ovr_classifier = OneVsRest(classifier=svm)

# to create a pipeline wityh various stages:

pipeline = Pipeline(stages=[indexer, tokenizer, stopwords_remover, hashingTF, idf, selector, normalizer, ovr_classifier])

# for splitting the data into training, validation, and test sets:

(training_data, validation_data, test_data) = data.randomSplit([0.7, 0.2, 0.1], seed=12345)

# to define parameter grid for grid search (regularization parameter (3 values), standardization of training features (2 values), and maximum number of iterations (2 values)):

param_grid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [500, 2000]) \
    .addGrid(svm.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(svm.maxIter, [10, 20]) \
    .build()

# evaluator for Multiclass Classification (F1 score):

evaluator = MulticlassClassificationEvaluator(metricName="f1")

# to perform grid search manually:

for params in param_grid:
    model = pipeline.copy(params).fit(training_data)
    predictions = model.transform(validation_data)
    f1_score = evaluator.evaluate(predictions)
    print(f"Parameters: {params}, F1 Score on Validation Set: {f1_score}")

# to make predictions on test data:

    predictions_test = model.transform(test_data)
    f1_score_test = evaluator.evaluate(predictions_test)
    print(f"F1 Score on Test Set with Parameters {params}: {f1_score_test}")

sc.stop()
