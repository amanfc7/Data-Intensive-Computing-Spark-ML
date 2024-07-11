from pyspark import SparkContext
from pyspark.sql import SparkSession
import json
import re
from collections import defaultdict

# for initializing the Spark context and spark session:

sc = SparkContext(appName="ChiSquareRDD")
spark = SparkSession.builder.appName("ChiSquareRDD").getOrCreate()

# below is function for loading stopwords from stopwords.txt:

def loading_stopwords(stopwords_file):
    with open(stopwords_file, 'r') as f:
        stopwords = {line.strip() for line in f}
    return stopwords

# now, the function for loading and preprocessing steps of the dataset:

def preprocessing(line, stopwords):
    review = json.loads(line)
    reviewText = review['reviewText']
    category = review['category']
    
    words = re.split(r'[^\w]+', reviewText.lower())
    words = [word for word in words if len(word) > 1 and word.isalpha() and word not in stopwords]   # to filter words only containing characters
    return (category, words)

# it will load stopwords from stopwords.txt:

stopwords_file = "src/stopwords.txt"
stopwords = loading_stopwords(stopwords_file)


data = sc.textFile("hdfs:///user/dic24_shared/amazon-reviews/full/reviews_devset.json")     # to use the reduced dataset for development
rdd = data.map(lambda line: preprocessing(line, stopwords)).cache()

# function for calculation of the chi-square values:

def chi_square(category_word_counts, total_counts, category_counts, total_docs):
    chi_square_value = defaultdict(float)
    for category, words in category_word_counts.items():
        for word, count in words.items():
            A = count
            B = category_counts[category] - count
            C = total_counts[word] - A
            N = total_docs
            D = N - (A + B + C)
            numerator = N * (A * D - B * C) ** 2
            denominator = (A + B) * (C + D) * (A + C) * (B + D)
            if denominator != 0:
                chi_square_value[(category, word)] = numerator / denominator
    return chi_square_value

# collecting category_word_count, total_count, category_count, total_docs using RDD's:

category_word_count = rdd.flatMapValues(lambda words: words) \
                          .map(lambda x: ((x[0], x[1]), 1)) \
                          .reduceByKey(lambda x, y: x + y) \
                          .map(lambda x: (x[0][0], {x[0][1]: x[1]})) \
                          .reduceByKey(lambda x, y: {**x, **y}) \
                          .collectAsMap()

total_count = rdd.flatMap(lambda x: set(x[1])) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda x, y: x + y) \
                  .collectAsMap()

category_count = rdd.map(lambda x: (x[0], 1)) \
                     .reduceByKey(lambda x, y: x + y) \
                     .collectAsMap()

total_docs = rdd.count()

# computing the chi-square value:

chi_square_value = chi_square(category_word_count, total_count, category_count, total_docs)

# now, to obtain top 75 terms for each category based on the chi-square values:

top_terms_per_category = {}
for category in category_word_count:
    sorted_terms = sorted([(word, chi_square_value[(category, word)]) for word in category_word_count[category]],
                          key=lambda x: -x[1])
    top_terms_per_category[category] = sorted_terms[:75]

# to obtain top terms from all categories:

top_terms = set()
for terms in top_terms_per_category.values():
    top_terms.update(term[0] for term in terms)

# creating a joined dictionary and for the output:

joined_dictionary = sorted(top_terms)

output_content = []    # to prepare the output 


# for writing the top terms per category:

for category in sorted(top_terms_per_category.keys()):
    terms = top_terms_per_category[category]
    output_content.append(f"Category: {category}")
    for term in terms:
        output_content.append(f"{term[0]}: {term[1]:.4f}")   
    output_content.append("\n")
    

# for seaparating and leaving some space between the chi square values of terms and the dictionary:

output_content.append("=" * 100)
output_content.append("DICTIONARY:")
output_content.append("=" * 100)

# to add the joined dictionary in the output file:

output_content.append(" ".join(joined_dictionary))

# for creating the output file with all the results:

output_file = "src/output_rdd.txt"
with open(output_file, "w") as f:
    for line in output_content:
        f.write(line + "\n")

sc.stop()
