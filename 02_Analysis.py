# Databricks notebook source
# MAGIC %md
# MAGIC # NLP Analysis of Game Reviews

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Initiate Config Variables

# COMMAND ----------

# MAGIC %run "./config/notebook_config"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 0a. Initialize variables and read in data

# COMMAND ----------

chunk_name_list = ['server','servers']

# COMMAND ----------

# Set catalog
_ = spark.sql(f"USE CATALOG {catalog_name}")

reviewsDF_filtered = spark.sql(f"SELECT * FROM {database_name}.steam_reviews_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Start some prep for NLP

# COMMAND ----------

# MAGIC %md
# MAGIC According to [PyPl](https://pypi.org/project/demoji/) we can use demoji to: "Accurately find or remove emojis from a blob of text using data from the Unicode Consortium's emoji code repository."
# MAGIC
# MAGIC This is useful because we can not only remove emojis, but keep some meaning with them for sentiment analysis.

# COMMAND ----------

# Define various text cleaning steps

# Unicode Normalization
def normalizeUnicode(text):
  return uni.normalize('NFKD',text).encode('ascii', 'ignore').decode('ascii')

# Handling Emoticons
def handleEmoticons(text):
  emojis = demoji.findall(text)
  for emoji in emojis:
    return text.replace(emoji, " " + emojis[emoji].split(":")[0])

# COMMAND ----------

# MAGIC %md
# MAGIC Apply the functions above to further cleanup on unicode and emoticons.

# COMMAND ----------

normalizeUnicodeUDF = udf(lambda text : normalizeUnicode(text), StringType())
handleEmoticonsUDF = udf(lambda text : handleEmoticons(text), StringType())

reviewsDF_filtered = reviewsDF_filtered.withColumn("reviewsNormalizeUnicode", normalizeUnicodeUDF(col("review")))
reviewsDF_filtered = reviewsDF_filtered.withColumn("reviewsCleaned", normalizeUnicodeUDF(col("reviewsNormalizeUnicode")))

reviewsDF_filtered_cleaned = reviewsDF_filtered.select("reviewsCleaned", *author_cols, "game_name")
display(reviewsDF_filtered_cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC [Spark NLP](https://nlp.johnsnowlabs.com/) is a state-of-the-art natural language processing library. This can also be installed using the Cluster Configuration UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Process the reviews and generate topics and sentiments

# COMMAND ----------

# MAGIC %md
# MAGIC <b>This is important:</b><br>
# MAGIC In order to run this successfully, [install this Maven package](https://docs.databricks.com/en/libraries/package-repositories.html#maven-or-spark-package) on the cluster:<br><br>
# MAGIC *com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0*

# COMMAND ----------

# MAGIC %md
# MAGIC This pipeline prepares, normalizes, tokenizes, and ultimately pulls out aspect based sentiment analysis. Further comments about each step are in the code block below.

# COMMAND ----------

# MAGIC %md
# MAGIC Reference: [Aspect Based Sentiment Analysis in Spark NLP](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ABSA_Inference.ipynb)

# COMMAND ----------

# Prepares data into a format that is processable by Spark NLP. This is the entry point for every Spark NLP pipeline
document_assembler = DocumentAssembler() \
    .setInputCol('reviewsCleaned')\
    .setOutputCol('document')\
    .setCleanupMode('each_full')

documentNormalizer = DocumentNormalizer() \
    .setInputCols("document") \
    .setOutputCol("normalizedDocument") \
    .setAction("clean") \
    .setPatterns(["\[.*?\]"]) \
    .setReplacement(" ") \
    .setPolicy("pretty_all") \
    .setLowercase(True)

# Annotator that detects sentence boundaries using regular expressions
sentence_detector = SentenceDetector() \
    .setInputCols(['normalizedDocument'])\
    .setOutputCol('sentence')

#Tokenizes raw sentences in sentence type columns into Tokens
tokenizer = Tokenizer()\
    .setInputCols(['sentence']) \
    .setOutputCol('token')

# Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary
normalizer = Normalizer() \
     .setInputCols(['token']) \
     .setOutputCol('normalizedTokens') \
     .setLowercase(True)

# Find lemmas out of words with the objective of returning a base dictionary word
lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalizedTokens']) \
     .setOutputCol('lemmaTokens')

# This annotator takes a sequence of strings (e.g. the output of a Tokenizer, Normalizer, Lemmatizer, and Stemmer) and drops all the stop words from the input sequences
stopwords_cleaner = StopWordsCleaner().pretrained() \
     .setInputCols(['lemmaTokens'])\
     .setOutputCol('cleanTokens')\
     .setCaseSensitive(False) 

# Word Embeddings lookup annotator that maps tokens to vectors(using Glove embedding here)
word_embeddings = WordEmbeddingsModel.pretrained("glove_6B_300", "xx")\
    .setInputCols(["normalizedDocument", "cleanTokens"])\
    .setOutputCol("embeddings")

# This Named Entity recognition annotator allows to train generic NER model based on Neural Networks. We are using a pre-trained model from John Snow Labs
ner_model = NerDLModel.pretrained("ner_aspect_based_sentiment")\
    .setInputCols(["normalizedDocument", "cleanTokens", "embeddings"])\
    .setOutputCol("ner")

# Converts a IOB or IOB2 representation of NER to a user-friendly one, by associating the tokens of recognized entities and their label
ner_converter = NerConverter() \
    .setInputCols(['sentence', 'cleanTokens', 'ner']) \
    .setOutputCol('ner_chunk')

# Using Pipeline functionality from Pyspark ML lib
nlp_pipeline = Pipeline(stages=[
    document_assembler, 
    documentNormalizer,
    sentence_detector,
    tokenizer,
    normalizer,
    lemmatizer,
    stopwords_cleaner,
    word_embeddings,
    ner_model,
    ner_converter])

# COMMAND ----------

# MAGIC %md
# MAGIC One of the best benefits of running ML pipelines in Databricks is having managed MLFlow. The model below gets automatically logged as an experiment.

# COMMAND ----------

try:
  mlflow.set_experiment(f"/Users/{useremail}/gaming_nlp_experiment") # will try creating experiment if it doesn't exist; but when two notebooks with this code executes at the same time, could trigger a race-condition
except:
  pass

# COMMAND ----------

result = nlp_pipeline.fit(reviewsDF_filtered_cleaned).transform(reviewsDF_filtered_cleaned)

# COMMAND ----------

result.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Here we cache the result from the model so that this result only gets run once. The results will then be in memory so that they can quickly be accessed for the following steps.

# COMMAND ----------

result.cache()
display(result.select("normalizedDocument.result"))

# COMMAND ----------

display(result.select("ner_chunk.result"))

# COMMAND ----------

exploded = F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata'))
select_expression_0 = F.expr("cols['result']").alias("chunk")
select_expression_1 = F.expr("cols['metadata']['entity']").alias("ner_label")
finalResults = result \
                  .select('reviewsCleaned',exploded.alias("cols"), *author_cols, "game_name") \
                  .select('reviewsCleaned',select_expression_0, select_expression_1, *author_cols, "game_name")
display(finalResults)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's visually inspect some of the results. Since server reviews are pretty common, we'll check for server and servers chunks.

# COMMAND ----------

analyzeManagerAspect = finalResults.filter(finalResults.chunk.isin(chunk_name_list))
display(analyzeManagerAspect)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Output data to Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC We can now write these results to [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html). We will use a catalog named <i>gaming_nlp_{username_sql}</i>, as defined in the config notebook. The catalog gets set as default further up in this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC One way to organize results is by having a schema for each game, here our schema is called <i>game_analysis</i>.

# COMMAND ----------

# Replace blanks in the name with underscore
#game_name_sub = re.sub(r"\s", "_", f"{game_name}")

# To prevent a table exists error, drop the table if it exists
_ = spark.sql(f"DROP TABLE IF EXISTS {database_name}.steam_reviews_silver")

# Use "delta" format for Unity Catalog
finalResults.write \
    .format("delta") \
    .saveAsTable(f"{database_name}.steam_reviews_silver")

# COMMAND ----------


