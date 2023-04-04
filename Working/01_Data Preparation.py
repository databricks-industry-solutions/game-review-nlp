# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Define functions to retrieve reviews

# COMMAND ----------

import requests

# COMMAND ----------

# MAGIC %md
# MAGIC Create a function that hits the steam API and returns results.<br><br><b>Note:</b> Might need to add more detail, maybe Duncan?

# COMMAND ----------

#@Anil - might be helpful to add a few comments here on what's going on
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

def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Use functions and REST API to get reviews

# COMMAND ----------

# MAGIC %md
# MAGIC Use [Steam REST API](https://partner.steamgames.com/doc/webapi_overview#2) to get reviews. <br><br><b>Note:</b> Add note on how to get App ID? Duncan?

# COMMAND ----------

# review_id = {"Football Manager":'1904540',
#              "New World":'1063730',
#              "Lost Ark":'1599340',
#              "Sonic Frontiers":'1237320'}

# print(review_id["New World"])

# COMMAND ----------

#reviews = get_n_reviews('1237320') # Sonic Frontiers
#reviews = get_n_reviews('1599340')
#reviews = get_n_reviews('1904540') # Sega football manager 2023

# COMMAND ----------

review_id = {"Football Manager":'1904540',
             "New World":'1063730',
             "Lost Ark":'1599340',
             "Sonic Frontiers":'1237320'}

reviews = get_n_reviews(review_id['New World']) # New World

# COMMAND ----------

#display(reviews)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Clean and subset data

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have reviews, let's reduce down to an actionable subset and create a Spark DataFrame.

# COMMAND ----------

from pyspark.sql.types import *

custom_schema = StructType([
          StructField("author", MapType(StringType(),StringType()), True),
#           StructField("author", StringType(), True),
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

#display(reviewsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, filter down to English reviews where the review is not null.

# COMMAND ----------

reviewsDF_filtered = reviewsDF \
                     .filter((reviewsDF.language=='english') & (reviewsDF.review.isNotNull())) \
                     .select('review','author')
#display(reviewsDF_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC Explode out author columns. This data will be valuable in segmenting reviews based on types of users to target different audiences.

# COMMAND ----------

from pyspark.sql.functions import col

reviewsDF_filtered = reviewsDF_filtered \
        .select(col("review"),
                col("author.steamid").alias("author_steamid"),
                col("author.num_reviews").alias("author_num_reviews").cast("int"),
                col("author.num_games_owned").alias("author_num_games_owned").cast("int"),
                col("author.last_played").alias("author_last_played").cast("int"),
                col("author.playtime_forever").alias("author_playtime_forever").cast("int"),
                col("author.playtime_at_review").alias("author_playtime_at_review").cast("int"),
                col("author.playtime_last_two_weeks").alias("author_playtime_at_last_two_weeks").cast("int")
               )

# COMMAND ----------

author_cols = ["author_steamid", "author_num_reviews", "author_num_games_owned", \
               "author_last_played", "author_playtime_forever", \
               "author_playtime_at_review", "author_playtime_at_last_two_weeks" \
              ]
#display(reviewsDF_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC Output:
# MAGIC * *reviewsDF_filtered*
# MAGIC * *author_cols*

# COMMAND ----------


