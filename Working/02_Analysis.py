# Databricks notebook source
# MAGIC %md
# MAGIC ## 0. Initiate Data Prep from 00 Notebook

# COMMAND ----------

# MAGIC %run "../Working/01_Data Preparation"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Start some prep for NLP

# COMMAND ----------

# MAGIC %md
# MAGIC According to [PyPl](https://pypi.org/project/demoji/) we can use demoji to: "Accurately find or remove emojis from a blob of text using data from the Unicode Consortium's emoji code repository."
# MAGIC 
# MAGIC This is useful because we can not only remove emojis, but keep some meaning with them for sentiment analysis.

# COMMAND ----------

#You can also install this by going to the Cluster Configuration > Libraries > Install New
!pip install demoji

# COMMAND ----------

#Unicodedata module provides access to the Unicode Character Database (UCD)
import unicodedata as uni

#Demoji helps to process emojis
import demoji

#re for regular expressions
import re

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
# MAGIC Further Cleanup on unicode and emoticons

# COMMAND ----------

from pyspark.sql.functions import col,udf
from pyspark.sql.types import StringType

normalizeUnicodeUDF = udf(lambda text : normalizeUnicode(text), StringType())
handleEmoticonsUDF = udf(lambda text : handleEmoticons(text), StringType())

reviewsDF_filtered = reviewsDF_filtered.withColumn("reviewsNormalizeUnicode", normalizeUnicodeUDF(col("review")))
reviewsDF_filtered = reviewsDF_filtered.withColumn("reviewsCleaned", normalizeUnicodeUDF(col("reviewsNormalizeUnicode")))

reviewsDF_filtered_cleaned = reviewsDF_filtered.select("reviewsCleaned", *author_cols)
display(reviewsDF_filtered_cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC [Spark NLP](https://nlp.johnsnowlabs.com/) is a state-of-the-art natural language processing library. This can also be installed using the Cluster Configuration UI.

# COMMAND ----------

# Install PySpark and Spark NLP
! pip install spark-nlp==4.2.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Process the reviews and generate topics and sentiments

# COMMAND ----------

# MAGIC %md
# MAGIC In order to run this successfully, intall this Maven package on the cluster:<br><br>
# MAGIC *com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0*

# COMMAND ----------

from pyspark.ml import Pipeline
import pyspark.sql.functions as F

import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *

# Reference - Aspect Based Sentiment Analysis in Spark NLP : https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ABSA_Inference.ipynb

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

result = nlp_pipeline.fit(reviewsDF_filtered_cleaned).transform(reviewsDF_filtered_cleaned)

# COMMAND ----------

result.printSchema()

# COMMAND ----------

display(result.select("normalizedDocument.result"))

# COMMAND ----------

display(result.select("ner_chunk.result"))

# COMMAND ----------

exploded = F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata'))
select_expression_0 = F.expr("cols['result']").alias("chunk")
select_expression_1 = F.expr("cols['metadata']['entity']").alias("ner_label")
finalResults = result \
                  .select('reviewsCleaned',exploded.alias("cols"), *author_cols) \
                  .select('reviewsCleaned',select_expression_0, select_expression_1, *author_cols)
display(finalResults)

# COMMAND ----------

# MAGIC %md
# MAGIC This is currently hard coded for football manager, let's either remove this or code it to be relevant to the game being queried.

# COMMAND ----------

# Visually inspect some of the results
analyzeManagerAspect = finalResults.filter((finalResults.chunk=='manager') | (finalResults.chunk=="football manager"))
display(analyzeManagerAspect)

# COMMAND ----------

analyzeManagerAspect.head(2)
