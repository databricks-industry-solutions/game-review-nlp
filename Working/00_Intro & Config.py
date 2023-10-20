# Databricks notebook source
# MAGIC %md
# MAGIC # Sentiment Analysis from [Steam](https://store.steampowered.com/) Game Reviews

# COMMAND ----------

# MAGIC %md
# MAGIC ![Steam Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Steam_icon_logo.svg/240px-Steam_icon_logo.svg.png)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This set of notebooks goes through 4 steps: Data Preparation, Analysis (NLP), Post Processing, and a Dashboard to extract insights from natural language reviews on Steam.
# MAGIC
# MAGIC 1. <i>Data Preparation:</i> This notebook connects to the Steam API and returns a DataFrame with reviews and data about the author.
# MAGIC 2. <i>Analysis:</i> This notebook runs a NLP pipeline to extract topics and sentiments. This notebook could be replaced with a different algorithm as well.
# MAGIC * In order to run this successfully, intall this Maven package on the cluster: *com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0*
# MAGIC * You can do this by opening your Cluster > Libraries Tab > Install New > Library Source: Maven > Paste the coordinates: com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0
# MAGIC 3. <i>Post Processing:</i>

# COMMAND ----------


