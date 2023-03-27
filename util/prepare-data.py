# Databricks notebook source
# MAGIC %md
# MAGIC #### Get Steam Reviews

# COMMAND ----------

# MAGIC %run ./notebook-config

# COMMAND ----------

import requests

# COMMAND ----------

def get_reviews(appid, params={'json':1}):
  url = 'https://store.steampowered.com/appreviews/'
  response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
  return response.json()

# COMMAND ----------

def get_n_reviews(appid, n=5000):
  reviews = []
  cursor = '*'
  params = {
    'json' : 1,
    'filter' : 'all',
    'language' : 'english',
    'day_range' : 9223372036854775807,
    'review_type' : 'all',
    'purchase_type' : 'all'
  }

  while n > 0:
    params['cursor'] = cursor.encode()
    params['num_per_page'] = min(100, n)
    n -= 100

    response = get_reviews(appid, params)
    cursor = response['cursor']
    reviews += response['reviews']

    if len(response['reviews']) < 100: break

  return reviews

# COMMAND ----------

game_dict = {'Sega Football Manager 2023': '1904540', 'Sonic Frontiers': '1237320', 'New World': '1063730'}

# COMMAND ----------

reviews = get_n_reviews(game_dict['New World'])

# COMMAND ----------

from pyspark.sql.types import *

custom_schema = StructType([
          StructField("author", MapType(StringType(),StringType()), True),
          StructField("comment_count", LongType(), True),
          StructField("language", StringType(), True),
          StructField("received_for_free", BooleanType(), True),
          StructField("recommendationid", StringType(), True),
          StructField("review", StringType(), True),
          StructField("steam_purchase", BooleanType(), True),
          StructField("timestamp_created", LongType(), True),
          StructField("timestamp_updated", LongType(), True),
          StructField("voted_up", BooleanType(), True),
          StructField("votes_funny", LongType(), True),
          StructField("votes_up", LongType(), True),
          StructField("weighted_vote_score", StringType(), True),
          StructField("written_during_early_access", BooleanType(), True),
        ])

reviewsDF = spark.createDataFrame(data=reviews, schema=custom_schema)

# COMMAND ----------

display(reviewsDF.select('review'))

# COMMAND ----------

reviewsDF.write.mode("overwrite").saveAsTable("bronze_reviews")

# COMMAND ----------


