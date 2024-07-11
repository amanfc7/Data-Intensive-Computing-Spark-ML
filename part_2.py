from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, ChiSqSelector, StringIndexer

# to initialize Spark context:

sc = SparkContext(appName="ChiSquareDF")
spark = SparkSession.builder.appName("ChiSquareDF").getOrCreate()

# for loading the stopwords:

stopwords = set()
with open("src/stopwords.txt", "r") as f:
    for line in f:
        stopwords.add(line.strip())

# now, function to load and preprocess the dataset:

def load_and_preprocess_data(file_path):
    data = spark.read.json(file_path)
    # Normalize text
    data = data.withColumn("text", lower(col("reviewText")))
    data = data.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))
    return data.select(col("category"), col("text"))

# loading data from the amazon review dataset for development:

data = load_and_preprocess_data("hdfs:///user/dic24_shared/amazon-reviews/full/reviews_devset.json")

# to define the structure and different stages of the pipeline:

indexer = StringIndexer(inputCol="category", outputCol="label")
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=list(stopwords))
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=2000)
idf = IDF(inputCol="raw_features", outputCol="features")
selector = ChiSqSelector(numTopFeatures=2000, featuresCol="features", outputCol="selected_features", labelCol="label")

# Creating of a pipeline:

pipeline = Pipeline(stages=[indexer, tokenizer, stopwords_remover, vectorizer, idf, selector])

# for fitting the pipeline to the given data:

model = pipeline.fit(data)

# for transforming the data:

result = model.transform(data)

# to extract the selected features (terms/words) from the data:

selected_indices = model.stages[-1].selectedFeatures
vocab = model.stages[3].vocabulary
selected_terms = [vocab[i] for i in selected_indices]

# Collection of the TF-IDF vectors for the first 2000 terms:

tfidf_vectors_rdd = result.select("selected_features").rdd.map(lambda row: row.selected_features.toArray())
tfidf_vectors = tfidf_vectors_rdd.take(2000)  # Limiting to 2000 terms for memory efficiency

# writing and finally saving the selected terms and their corresponding TF-IDF vectors to a file:

output_file = "src/output_ds.txt"
with open(output_file, "w") as f:
    for term, vec in zip(selected_terms, tfidf_vectors):
        f.write(f"{term}: {vec}\n")


sc.stop()    # to stop the Spark context
