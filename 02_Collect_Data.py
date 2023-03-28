# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/game-review-nlp. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/game-review-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC ### In this notebook you:
# MAGIC * Create a database for the tables to reside in.
# MAGIC * Retrieve data via an API and store it in a `Delta` table.
# MAGIC * Clean and transform the retrieved data.
# MAGIC * Create tables for easy access and queriability.
# MAGIC * Explore the dataset and relationships.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Setup notebook configuration

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

import requests
from delta import DeltaTable
from typing import Tuple
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create tables

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS steam_applist_bronze (
# MAGIC   appid INT NOT NULL,
# MAGIC   name STRING NOT NULL) 
# MAGIC TBLPROPERTIES (
# MAGIC   quality = 'bronze',
# MAGIC   delta.enableChangeDataFeed = true, 
# MAGIC   delta.autoOptimize.autoCompact = true);
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS steam_appreviews_bronze (
# MAGIC   recommendationid LONG,
# MAGIC   appid INT,
# MAGIC   timestamp_created LONG,
# MAGIC   timestamp_updated LONG,
# MAGIC   author MAP<STRING,STRING>,
# MAGIC   language STRING,
# MAGIC   review STRING,
# MAGIC   voted_up BOOLEAN,
# MAGIC   votes_up INT,
# MAGIC   votes_funny INT,
# MAGIC   weighted_vote_score DOUBLE,
# MAGIC   comment_count INT,
# MAGIC   steam_purchase BOOLEAN,
# MAGIC   received_for_free BOOLEAN,
# MAGIC   written_during_early_access BOOLEAN,
# MAGIC   hidden_in_steam_china BOOLEAN,
# MAGIC   steam_china_location STRING)
# MAGIC TBLPROPERTIES (
# MAGIC   quality = 'bronze',
# MAGIC   delta.enableChangeDataFeed = true,
# MAGIC   delta.autoOptimize.optimizeWrite = true, 
# MAGIC   delta.autoOptimize.autoCompact = true);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Collect data into bronze layer from the Steam web API
# MAGIC 
# MAGIC First, we'll collect an applist, so we have a set of appid's and names
# MAGIC for all apps currently on the Steam store. For this accelerator, we'll
# MAGIC only download the reviews for one, or maybe a few apps. This could
# MAGIC be extended downstream in silver or gold with additional information
# MAGIC for other apps, and in the meantime will be useful for quickly looking
# MAGIC up the appid of a title for which you want reviews downloaded.

# COMMAND ----------

def update_applist_bronze(applist_bronze_table_name: str) -> None:
    """Update the app reviews table."""
    url = 'https://api.steampowered.com/ISteamApps/GetAppList/v2/'
    response = requests.get(url)
    apps = response.json()['applist']['apps']
    src = spark.createDataFrame(apps).alias('src')
    dst = DeltaTable.forName(spark, applist_bronze_table_name).alias('dst')
    merge_builder = (
        dst.merge(src, 'dst.appid = src.appid')
        .whenNotMatchedInsertAll())
    merge_builder.execute()

applist_bronze_table_name = 'steam_applist_bronze'
update_applist_bronze(applist_bronze_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Next we need to download the reviews themselves. The web API provided
# MAGIC by Steam makes this relatively easy; however, we need to download records
# MAGIC page by page using a cursor, with the maximum page size being 100 reviews.
# MAGIC Since that's not many records to use for a single merge, we download them
# MAGIC into a batch, and then insert the entire batch of 100 pages at one time.
# MAGIC 
# MAGIC If something happens with your download, such as a temporary error, the
# MAGIC cursor is returned so you can resume from there if you are backfilling.
# MAGIC 
# MAGIC **Note: For DotA 2, there are over 600k reviews. In testing, this takes
# MAGIC about a minute per batch of 100 pages, so you can expect this process to 
# MAGIC take roughly an hour to backfill the entire dataset.**

# COMMAND ----------

def update_appreviews_bronze(
    reviews_table_name: str, 
    appid: int, 
    start_cursor: str = '*', 
    backfill: bool = False, 
    batches_per_insert: int = 100
) -> Tuple[str, str]:
    """Update the appreviews bronze table."""
    dst = DeltaTable.forName(spark, reviews_table_name).alias('dst')
    url = f'https://store.steampowered.com/appreviews/{appid}'

    # If we're not backfilling, we need to figure out how far back to go,
    # so we check for the latest created timestamp we've loaded so far.
    if not backfill:
        max_timestamp_created = (
            dst.toDF()
            .filter(F.col('appid') == appid)
            .groupBy()
            .max('timestamp_created')
            .first()[0])

    # Prepare the data needed for the request. You can read more about these here:
    # https://partner.steamgames.com/doc/store/getreviews
    payload = {
        'filter': 'recent',
        'language': 'english',
        'cursor': None,
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page': 100,
        'json': 1 }

    cursor = start_cursor
    done = False

    # Load batches until we reach a stopping condition
    # Here, batching is used to control the number of merges.
    # The API only allows us to download 100 rows per call, which
    # is a relatively small number to merge each iteration.
    while not done:
        batch = []  # reset the batch
        for i in range(batches_per_insert):
            # Call the API for the current cursor to download a page of reviews.
            payload['cursor'] = cursor
            response = requests.get(url, params=payload)
            data = response.json()

            # Check for either a bad batch, or completion (i.e., most recent 
            # cursor returned no rows). In either case, we don't have another
            # set to extend, so break and write whatever batch we have so far.
            if data['success'] != 1:
                status = 'failed'
                done = True
                break
            elif data['query_summary']['num_reviews'] == 0:
                status = 'complete'
                done = True
                break

            # We collected another set of reviews, so we can extend the batch,
            # and update the cursor.
            reviews = data['reviews']
            for review in reviews:
                review['appid'] = appid
            batch.extend(reviews)
            cursor = data['cursor']

            # Unless we're backfilling, check another stop condition based
            # on whether we've started getting into reviews created which are
            # older than ones we've already collected for this app.
            if not backfill:
                latest_timestamp = reviews[-1]['timestamp_created']
                if (max_timestamp_created is not None and latest_timestamp < max_timestamp_created):
                    status = 'caught up'
                    done = True
                    break
        
        # Merge in this batch of records.
        src = spark.createDataFrame(batch).alias('src')
        cond = 'dst.recommendationid = src.recommendationid'
        dst.merge(src, cond).whenNotMatchedInsertAll().execute()

    return status, cursor

# DotA 2: 570
# New World: 1063730
appid = 1063730

appreviews_bronze_table_name = 'steam_appreviews_bronze'
status, cursor = update_appreviews_bronze(appreviews_bronze_table_name, appid, backfill=True)
print(status, cursor)
