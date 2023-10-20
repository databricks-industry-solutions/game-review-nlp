# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation for Game Reviews from Steam

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook connects to the Steam API and returns a DataFrame of reviews for a particular game. The schema returned includes a text review as well as data about the author, the timestamp, and amount of time the author has spent in the game. We will use this data to parse out insights from the data as well as to group results in the dashboard by author groups.
# MAGIC
# MAGIC First we define functions to connect to [the Steam API](https://partner.steamgames.com/doc/webapi_overview#2).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define functions to retrieve reviews

# COMMAND ----------

import requests

# COMMAND ----------

# MAGIC %md
# MAGIC The first step is to create two functions. The first (<i>get_n_reviews</i>) controls the number of reviews and the number per page.

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

# MAGIC %md
# MAGIC The second (<i>get_reviews</i>) sets the URL and makes the request to the API with the parameters that come from <i>get_n_reviews</i>.

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
# MAGIC Use [Steam REST API](https://partner.steamgames.com/doc/webapi_overview#2) to get reviews. 
# MAGIC <br>
# MAGIC <br>
# MAGIC <b>Note:</b> You can get an app ID by looking up a game on [the Steam website](https://store.steampowered.com/) and getting the ID from the URL. In the <i>review_id</i> dictionary below we include a few IDs.

# COMMAND ----------

review_id = {"Sega Football Manager":'1904540',
             "New World":'1063730',
             "Lost Ark":'1599340',
             "Sonic Frontiers":'1237320'}

reviews = get_n_reviews(review_id['New World']) # New World

# COMMAND ----------

# MAGIC %md
# MAGIC From here, we will be using reviews for [New World](https://store.steampowered.com/app/1063730/New_World/). To reduce output to the screen, we've commented out the <i>display</i> below, but if you're troubleshooting or want extra output, you can uncomment it.

# COMMAND ----------

#display(reviews)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Clean and subset data

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

from pyspark.sql.functions import col, from_unixtime

reviewsDF_filtered = reviewsDF_filtered \
        .select(col("review"),
                col("author.steamid").alias("author_steamid"),
                col("author.num_reviews").alias("author_num_reviews").cast("int"),
                col("author.num_games_owned").alias("author_num_games_owned").cast("int"),
                from_unixtime(col("author.last_played")).alias("author_last_played").cast("timestamp"),
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
# MAGIC The <i>01_Data Preparation</i> notebook outputs:
# MAGIC * A DataFrame called *reviewsDF_filtered* which has: *review*, and *author_cols*
# MAGIC * A Python list called *author_cols* that lists the columns related to authors

# COMMAND ----------


