# Databricks notebook source
# MAGIC %md
# MAGIC <a href="https://www.databricks.com/solutions/accelerators"><img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png'></img></a>
# MAGIC
# MAGIC
# MAGIC The series of assets inside this repo show a few of the capabilities possible when leveraging Databricks as the analytics platform of choice in your studio.

# COMMAND ----------

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
# MAGIC * In order to run this successfully, install this Maven package on the cluster: *com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0*
# MAGIC * You can do this by opening your Cluster > Libraries Tab > Install New > Library Source: Maven > Paste the coordinates: com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0
# MAGIC * Note: If you use the Solution Accelerator cluster, it takes care of this for you.
# MAGIC 3. <i>Post Processing:</i> This notebook profiles and clusters the review authors, saving the results to a Delta table to be used in the dashboard.
# MAGIC 4. <i>Player Feedback Sentiment Analysis Dashboard</i>: The dashboard uses [Databricks Lakeview Dashboards](https://docs.databricks.com/en/dashboards/lakeview.html) and is provided in this bundle as a json file. In order to use it, you can import the dashboard by going to Dashboards > Lakeview Dashboards > Hit the down arrow next to "Create Lakeview Dashboard" > Import Lakeview Dashboard from File 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install and import necessary packages

# COMMAND ----------

# MAGIC %run "./config/notebook_config"

# COMMAND ----------

# Check if the catalog already exists
catalog_exists = spark.sql(f"SHOW CATALOGS LIKE '{catalog_name}'").count() > 0

# Create the catalog if it does not exist
if not catalog_exists:
    _ = spark.sql(f"CREATE CATALOG {catalog_name}")

# Set catalog
_ = spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# Check if database exists, if not create it
_ = spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")

# Set database
_ = spark.sql(f"USE {database_name}")

# COMMAND ----------

try:
  mlflow.set_experiment(f"/Users/{useremail}/gaming_nlp_experiment") # will try creating experiment if it doesn't exist; but when two notebooks with this code executes at the same time, could trigger a race-condition
except:
  pass

# COMMAND ----------

print(f"Username: {username}")
print(f"tmpdir: {tmpdir}")
print(f"Catalog Name: {catalog_name}")
print(f"Database Name: {database_name}")

# COMMAND ----------


